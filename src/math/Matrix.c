// https://github.com/alexjasson/tensorgrid

#include "Matrix.h"
#include <stddef.h>
#include "Vector.h"
#include "Scalar.h"

// An MR x NRV block of vector registers, used by microKernel
#define NRV 2
#define NR  (NRV * VECTOR_DIMS)
#define MR  ((VECTOR_CARD - NRV - 1) / NRV)

// Cache block sizes for the temporary packed A/B blocks
#define KC 256
#define NC (NR * 32)
#define MC (MR * 16)

// The unpacked small shape kernel is used when K and N are both small
#define SMALL_K 256
#define SMALL_N 256

static int imin(int a, int b);
static void packB(Real *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock);
static void packA(Real *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock);
static void microKernelContiguous(Real *Ap, Real *Bp, int kBlock, Matrix C);
static void microKernel(Real *Ap, Real *Bp, int kBlock, Matrix C, int mValid, int nValid);
static void smallGemm(Matrix A, Matrix B, Matrix C);
static void packedGemm(Matrix A, Matrix B, Matrix C);

Matrix MatrixView(Real *data, int rows, int cols, int sRow, int sCol)
{
    return (Matrix){data, rows, cols, sRow, sCol, REAL, 1.0, NULL};
}

static int imin(int a, int b) { return a < b ? a : b; }

// Pack a B block into NR-wide panels, padding the right edge with zeros
static void packB(Real *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock)
{
    int panels = (nBlock + NR - 1) / NR;
    int fullPanels = (B.sCol == 1) ? nBlock / NR : 0;

    if (B.type == REAL && B.scale == 1.0 && !B.scales) {
        // Vectorize contiguous rows
        for (int jp = 0; jp < fullPanels; jp++) {
            Real *dst = Bp + (size_t)jp * kBlock * NR;
            Real *src = B.data + (size_t)p0 * B.sRow + j0 + (size_t)jp * NR;
            for (int p = 0; p < kBlock; p++)
                for (int v = 0; v < NR; v += VECTOR_DIMS)
                    VectorStore(dst + p * NR + v, VectorLoad(src + (size_t)p * B.sRow + v, REAL));
        }

        for (int jp = fullPanels; jp < panels; jp++) {
            Real *dst = Bp + (size_t)jp * kBlock * NR;
            for (int p = 0; p < kBlock; p++) {
                Real *brow = B.data + (size_t)(p0 + p) * B.sRow;
                for (int nr = 0; nr < NR; nr++) {
                    int col = jp * NR + nr;
                    dst[p * NR + nr] = (col < nBlock) ? brow[(size_t)(j0 + col) * B.sCol] : 0.0f;
                }
            }
        }
        return;
    }

    size_t esz = ScalarSize(B.type);
    const unsigned char *base = (const unsigned char *)B.data;

    for (int jp = 0; jp < fullPanels; jp++) {
        Real *dst = Bp + (size_t)jp * kBlock * NR;
        const unsigned char *src = base + ((size_t)p0 * B.sRow + j0 + (size_t)jp * NR) * esz;
        for (int p = 0; p < kBlock; p++) {
            Vector scale = VectorBcast(B.scales ? B.scale * B.scales[p0 + p] : B.scale);
            for (int v = 0; v < NR; v += VECTOR_DIMS)
                VectorStore(dst + p * NR + v, VectorMul(VectorLoad(src + ((size_t)p * B.sRow + v) * esz, B.type), scale));
        }
    }

    for (int jp = fullPanels; jp < panels; jp++) {
        Real *dst = Bp + (size_t)jp * kBlock * NR;
        for (int p = 0; p < kBlock; p++) {
            const unsigned char *brow = base + (size_t)(p0 + p) * B.sRow * esz;
            Real s = B.scales ? B.scale * B.scales[p0 + p] : B.scale;
            for (int nr = 0; nr < NR; nr++) {
                int col = jp * NR + nr;
                dst[p * NR + nr] = (col < nBlock) ? s * ScalarLoad(brow + (size_t)(j0 + col) * B.sCol * esz, B.type) : 0.0f;
            }
        }
    }
}

// Pack an A block into MR-row panels, padding the bottom edge with zeros
static void packA(Real *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock)
{
    int panels = (mBlock + MR - 1) / MR;

    if (A.type == REAL && !A.scales) {
        for (int ip = 0; ip < panels; ip++) {
            Real *dst = Ap + (size_t)ip * kBlock * MR;
            for (int p = 0; p < kBlock; p++) {
                for (int mr = 0; mr < MR; mr++) {
                    int row = ip * MR + mr;
                    dst[p * MR + mr] = (row < mBlock) ? A.data[(size_t)(i0 + row) * A.sRow + (size_t)(p0 + p) * A.sCol] : 0.0f;
                }
            }
        }
        return;
    }

    size_t esz = ScalarSize(A.type);
    const unsigned char *base = (const unsigned char *)A.data;
    for (int ip = 0; ip < panels; ip++) {
        Real *dst = Ap + (size_t)ip * kBlock * MR;
        for (int p = 0; p < kBlock; p++) {
            for (int mr = 0; mr < MR; mr++) {
                int row = ip * MR + mr;
                dst[p * MR + mr] = (row < mBlock) ? (A.scales ? A.scales[i0 + row] : (Real)1.0) * ScalarLoad(base + ((size_t)(i0 + row) * A.sRow + (size_t)(p0 + p) * A.sCol) * esz, A.type) : 0.0f;
            }
        }
    }
}

static void microKernelContiguous(Real *Ap, Real *Bp, int kBlock, Matrix C)
{
    Vector acc[MR][NRV];
    for (int mr = 0; mr < MR; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VectorZero();

    for (int mr = 0; mr < MR; mr++) {
        VectorFetch(C.data + (size_t)mr * C.sRow);
        VectorFetch(C.data + (size_t)mr * C.sRow + NR - 1);
    }

    Real *aRow = Ap;
    Real *bRow = Bp;
    for (int p = 0; p < kBlock; p++, aRow += MR, bRow += NR) {
        Vector b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VectorLoad(bRow + v * VECTOR_DIMS, REAL);
        for (int mr = 0; mr < MR; mr++) {
            Vector a = VectorBcast(aRow[mr]);
            for (int v = 0; v < NRV; v++)
                acc[mr][v] = VectorFma(a, b[v], acc[mr][v]);
        }
    }

    for (int mr = 0; mr < MR; mr++) {
        Real *Crow = C.data + (size_t)mr * C.sRow;
        for (int v = 0; v < NRV; v++)
            VectorStore(Crow + v * VECTOR_DIMS, VectorAdd(VectorLoad(Crow + v * VECTOR_DIMS, REAL), acc[mr][v]));
    }
}

static void microKernel(Real *Ap, Real *Bp, int kBlock, Matrix C, int mValid, int nValid)
{
    Vector acc[MR][NRV];
    for (int mr = 0; mr < mValid; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VectorZero();

    for (int mr = 0; mr < mValid; mr++) {
        VectorFetch(C.data + (size_t)mr * C.sRow);
        VectorFetch(C.data + (size_t)mr * C.sRow + (size_t)(NR - 1) * C.sCol);
    }

    Real *aRow = Ap;
    Real *bRow = Bp;
    for (int p = 0; p < kBlock; p++, aRow += MR, bRow += NR) {
        Vector b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VectorLoad(bRow + v * VECTOR_DIMS, REAL);
        for (int mr = 0; mr < mValid; mr++) {
            Vector a = VectorBcast(aRow[mr]);
            for (int v = 0; v < NRV; v++)
                acc[mr][v] = VectorFma(a, b[v], acc[mr][v]);
        }
    }

    if (C.sCol == 1 && nValid == NR) {
        for (int mr = 0; mr < mValid; mr++) {
            Real *Crow = C.data + (size_t)mr * C.sRow;
            for (int v = 0; v < NRV; v++)
                VectorStore(Crow + v * VECTOR_DIMS, VectorAdd(VectorLoad(Crow + v * VECTOR_DIMS, REAL), acc[mr][v]));
        }
        return;
    }

    for (int mr = 0; mr < mValid; mr++) {
        Real tile[NR];
        for (int v = 0; v < NRV; v++)
            VectorStore(tile + v * VECTOR_DIMS, acc[mr][v]);
        for (int nr = 0; nr < nValid; nr++)
            C.data[mr * C.sRow + nr * C.sCol] += tile[nr];
    }
}

// GEMM (C += A @ B) for small shapes where packing overhead would dominate
static void smallGemm(Matrix A, Matrix B, Matrix C)
{
    int M = C.rows, N = C.cols, K = A.cols;
    int sA = A.sRow, sAc = A.sCol, sC = C.sRow;
    Real Bpack[SMALL_K * SMALL_N];
    const Real *Bd;
    int sB;

    B.scale = A.scale * B.scale / C.scale;
    if (B.sCol == 1 && B.type == REAL && B.scale == 1.0 && !B.scales) {
        Bd = B.data;
        sB = B.sRow;
    } else if (B.type == REAL && B.scale == 1.0 && !B.scales) {
        for (int p = 0; p < K; p++)
            for (int j = 0; j < N; j++)
                Bpack[(size_t)p * N + j] = B.data[(size_t)p * B.sRow + (size_t)j * B.sCol];
        Bd = Bpack;
        sB = N;
    } else {
        size_t esz = ScalarSize(B.type);
        const unsigned char *base = (const unsigned char *)B.data;
        for (int p = 0; p < K; p++) {
            Real s = B.scales ? B.scale * B.scales[p] : B.scale;
            for (int j = 0; j < N; j++)
                Bpack[(size_t)p * N + j] = s * ScalarLoad(base + ((size_t)p * B.sRow + (size_t)j * B.sCol) * esz, B.type);
        }
        Bd = Bpack;
        sB = N;
    }

    for (int i0 = 0; i0 < M; i0 += MR) {
        int mValid = imin(MR, M - i0);
        const Real *Ai = A.data + (size_t)i0 * sA;
        int j0 = 0;

        for (; j0 + NR <= N; j0 += NR) {
            const Real *Bj = Bd + j0;
            Vector acc[MR][NRV];
            for (int m = 0; m < mValid; m++)
                for (int v = 0; v < NRV; v++) acc[m][v] = VectorZero();

            if (mValid == MR) {
                const Real *Acol = Ai;
                const Real *Brow = Bj;
                for (int p = 0; p < K; p++, Acol += sAc, Brow += sB) {
                    Vector b[NRV];
                    for (int v = 0; v < NRV; v++) b[v] = VectorLoad(Brow + v * VECTOR_DIMS, REAL);
                    for (int m = 0; m < MR; m++) {
                        Vector a = VectorBcast(Acol[(size_t)m * sA]);
                        for (int v = 0; v < NRV; v++) acc[m][v] = VectorFma(a, b[v], acc[m][v]);
                    }
                }
            } else {
                const Real *Acol = Ai;
                const Real *Brow = Bj;
                for (int p = 0; p < K; p++, Acol += sAc, Brow += sB) {
                    Vector b[NRV];
                    for (int v = 0; v < NRV; v++) b[v] = VectorLoad(Brow + v * VECTOR_DIMS, REAL);
                    for (int m = 0; m < mValid; m++) {
                        Vector a = VectorBcast(Acol[(size_t)m * sA]);
                        for (int v = 0; v < NRV; v++) acc[m][v] = VectorFma(a, b[v], acc[m][v]);
                    }
                }
            }

            for (int m = 0; m < mValid; m++) {
                Real *Cm = C.data + (size_t)(i0 + m) * sC + j0;
                for (int v = 0; v < NRV; v++)
                    VectorStore(Cm + v * VECTOR_DIMS, VectorAdd(VectorLoad(Cm + v * VECTOR_DIMS, REAL), acc[m][v]));
            }
        }

        for (; j0 < N; j0++)
            for (int m = 0; m < mValid; m++) {
                Real s = 0;
                const Real *a = Ai + (size_t)m * sA;
                const Real *b = Bd + j0;
                for (int p = 0; p < K; p++, a += sAc, b += sB)
                    s += *a * *b;
                C.data[(size_t)(i0 + m) * sC + j0] += s;
            }
    }
}

// cache-blocked packed matrix multiply
static void packedGemm(Matrix A, Matrix B, Matrix C)
{
    int M = C.rows;
    int N = C.cols;
    int K = A.cols;

    Real packBufA[MC * KC];
    Real packBufB[NC * KC];

    B.scale = A.scale * B.scale / C.scale;

    for (int j0 = 0; j0 < N; j0 += NC) {
        int nBlock = imin(NC, N - j0);
        for (int p0 = 0; p0 < K; p0 += KC) {
            int kBlock = imin(KC, K - p0);
            packB(packBufB, B, p0, kBlock, j0, nBlock);
            for (int i0 = 0; i0 < M; i0 += MC) {
                int mBlock = imin(MC, M - i0);
                packA(packBufA, A, i0, mBlock, p0, kBlock);
                for (int jr = 0; jr < nBlock; jr += NR) {
                    Real *Bpanel = packBufB + (size_t)(jr / NR) * kBlock * NR;
                    for (int ir = 0; ir < mBlock; ir += MR) {
                        Real *Apanel = packBufA + (size_t)(ir / MR) * kBlock * MR;
                        Matrix Ctile = { C.data + (size_t)(i0 + ir) * C.sRow + (size_t)(j0 + jr) * C.sCol, 0, 0, C.sRow, C.sCol, REAL, 1.0, NULL };
                        int mValid = imin(MR, mBlock - ir);
                        int nValid = imin(NR, nBlock - jr);
                        if (C.sCol == 1 && mValid == MR && nValid == NR)
                            microKernelContiguous(Apanel, Bpanel, kBlock, Ctile);
                        else
                            microKernel(Apanel, Bpanel, kBlock, Ctile, mValid, nValid);
                    }
                }
            }
        }
    }
}

void MatrixMultiply(Matrix A, Matrix B, Matrix C)
{
    if (A.type == REAL && !A.scales && A.cols <= SMALL_K && C.cols >= NR && C.cols <= SMALL_N && C.sCol == 1)
        smallGemm(A, B, C);
    else
        packedGemm(A, B, C);
}
