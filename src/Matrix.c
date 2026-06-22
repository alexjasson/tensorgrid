// https://github.com/alexjasson/tensorgrid

#include "Matrix.h"
#include "Vector.h"

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
static void packB(Entry *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock);
static void packA(Entry *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock);
static void microKernelContiguous(Entry *Ap, Entry *Bp, int kBlock, Matrix C);
static void microKernel(Entry *Ap, Entry *Bp, int kBlock, Matrix C, int mValid, int nValid);
static void smallGemm(Matrix A, Matrix B, Matrix C);
static void packedGemm(Matrix A, Matrix B, Matrix C);

Matrix MatrixView(Entry *data, int rows, int cols, int sRow, int sCol)
{
    return (Matrix){data, rows, cols, sRow, sCol};
}

static int imin(int a, int b) { return a < b ? a : b; }

// Pack a B block into NR-wide panels, padding the right edge with zeros
static void packB(Entry *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock)
{
    int panels = (nBlock + NR - 1) / NR;
    int fullPanels = (B.sCol == 1) ? nBlock / NR : 0;

    // Vectorize contiguous rows
    for (int jp = 0; jp < fullPanels; jp++) {
        Entry *dst = Bp + (size_t)jp * kBlock * NR;
        Entry *src = B.data + (size_t)p0 * B.sRow + j0 + (size_t)jp * NR;
        for (int p = 0; p < kBlock; p++)
            for (int v = 0; v < NR; v += VECTOR_DIMS)
                VectorStore(dst + p * NR + v, VectorLoad(src + (size_t)p * B.sRow + v));
    }

    for (int jp = fullPanels; jp < panels; jp++) {
        Entry *dst = Bp + (size_t)jp * kBlock * NR;
        for (int p = 0; p < kBlock; p++) {
            Entry *brow = B.data + (size_t)(p0 + p) * B.sRow;
            for (int nr = 0; nr < NR; nr++) {
                int col = jp * NR + nr;
                dst[p * NR + nr] = (col < nBlock) ? brow[(size_t)(j0 + col) * B.sCol] : 0.0f;
            }
        }
    }
}

// Pack an A block into MR-row panels, padding the bottom edge with zeros
static void packA(Entry *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock)
{
    int panels = (mBlock + MR - 1) / MR;
    for (int ip = 0; ip < panels; ip++) {
        Entry *dst = Ap + (size_t)ip * kBlock * MR;
        for (int p = 0; p < kBlock; p++) {
            for (int mr = 0; mr < MR; mr++) {
                int row = ip * MR + mr;
                dst[p * MR + mr] = (row < mBlock) ? A.data[(size_t)(i0 + row) * A.sRow + (size_t)(p0 + p) * A.sCol] : 0.0f;
            }
        }
    }
}

static void microKernelContiguous(Entry *Ap, Entry *Bp, int kBlock, Matrix C)
{
    Vector acc[MR][NRV];
    for (int mr = 0; mr < MR; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VectorZero();

    for (int mr = 0; mr < MR; mr++) {
        VectorFetch(C.data + (size_t)mr * C.sRow);
        VectorFetch(C.data + (size_t)mr * C.sRow + NR - 1);
    }

    Entry *aRow = Ap;
    Entry *bRow = Bp;
    for (int p = 0; p < kBlock; p++, aRow += MR, bRow += NR) {
        Vector b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VectorLoad(bRow + v * VECTOR_DIMS);
        for (int mr = 0; mr < MR; mr++) {
            Vector a = VectorBcast(aRow[mr]);
            for (int v = 0; v < NRV; v++)
                acc[mr][v] = VectorFma(a, b[v], acc[mr][v]);
        }
    }

    for (int mr = 0; mr < MR; mr++) {
        Entry *Crow = C.data + (size_t)mr * C.sRow;
        for (int v = 0; v < NRV; v++)
            VectorStore(Crow + v * VECTOR_DIMS, VectorAdd(VectorLoad(Crow + v * VECTOR_DIMS), acc[mr][v]));
    }
}

static void microKernel(Entry *Ap, Entry *Bp, int kBlock, Matrix C, int mValid, int nValid)
{
    Vector acc[MR][NRV];
    for (int mr = 0; mr < mValid; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VectorZero();

    for (int mr = 0; mr < mValid; mr++) {
        VectorFetch(C.data + (size_t)mr * C.sRow);
        VectorFetch(C.data + (size_t)mr * C.sRow + (size_t)(NR - 1) * C.sCol);
    }

    Entry *aRow = Ap;
    Entry *bRow = Bp;
    for (int p = 0; p < kBlock; p++, aRow += MR, bRow += NR) {
        Vector b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VectorLoad(bRow + v * VECTOR_DIMS);
        for (int mr = 0; mr < mValid; mr++) {
            Vector a = VectorBcast(aRow[mr]);
            for (int v = 0; v < NRV; v++)
                acc[mr][v] = VectorFma(a, b[v], acc[mr][v]);
        }
    }

    if (C.sCol == 1 && nValid == NR) {
        for (int mr = 0; mr < mValid; mr++) {
            Entry *Crow = C.data + (size_t)mr * C.sRow;
            for (int v = 0; v < NRV; v++)
                VectorStore(Crow + v * VECTOR_DIMS, VectorAdd(VectorLoad(Crow + v * VECTOR_DIMS), acc[mr][v]));
        }
        return;
    }

    for (int mr = 0; mr < mValid; mr++) {
        Entry tile[NR];
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
    Entry Bpack[SMALL_K * SMALL_N];
    const Entry *Bd;
    int sB;

    if (B.sCol == 1) {
        Bd = B.data;
        sB = B.sRow;
    } else {
        for (int p = 0; p < K; p++)
            for (int j = 0; j < N; j++)
                Bpack[(size_t)p * N + j] = B.data[(size_t)p * B.sRow + (size_t)j * B.sCol];
        Bd = Bpack;
        sB = N;
    }

    for (int i0 = 0; i0 < M; i0 += MR) {
        int mValid = imin(MR, M - i0);
        const Entry *Ai = A.data + (size_t)i0 * sA;
        int j0 = 0;

        for (; j0 + NR <= N; j0 += NR) {
            const Entry *Bj = Bd + j0;
            Vector acc[MR][NRV];
            for (int m = 0; m < mValid; m++)
                for (int v = 0; v < NRV; v++) acc[m][v] = VectorZero();

            if (mValid == MR) {
                const Entry *Acol = Ai;
                const Entry *Brow = Bj;
                for (int p = 0; p < K; p++, Acol += sAc, Brow += sB) {
                    Vector b[NRV];
                    for (int v = 0; v < NRV; v++) b[v] = VectorLoad(Brow + v * VECTOR_DIMS);
                    for (int m = 0; m < MR; m++) {
                        Vector a = VectorBcast(Acol[(size_t)m * sA]);
                        for (int v = 0; v < NRV; v++) acc[m][v] = VectorFma(a, b[v], acc[m][v]);
                    }
                }
            } else {
                const Entry *Acol = Ai;
                const Entry *Brow = Bj;
                for (int p = 0; p < K; p++, Acol += sAc, Brow += sB) {
                    Vector b[NRV];
                    for (int v = 0; v < NRV; v++) b[v] = VectorLoad(Brow + v * VECTOR_DIMS);
                    for (int m = 0; m < mValid; m++) {
                        Vector a = VectorBcast(Acol[(size_t)m * sA]);
                        for (int v = 0; v < NRV; v++) acc[m][v] = VectorFma(a, b[v], acc[m][v]);
                    }
                }
            }

            for (int m = 0; m < mValid; m++) {
                Entry *Cm = C.data + (size_t)(i0 + m) * sC + j0;
                for (int v = 0; v < NRV; v++)
                    VectorStore(Cm + v * VECTOR_DIMS, VectorAdd(VectorLoad(Cm + v * VECTOR_DIMS), acc[m][v]));
            }
        }

        for (; j0 < N; j0++)
            for (int m = 0; m < mValid; m++) {
                Entry s = 0;
                const Entry *a = Ai + (size_t)m * sA;
                const Entry *b = Bd + j0;
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

    Entry packBufA[MC * KC];
    Entry packBufB[NC * KC];

    for (int j0 = 0; j0 < N; j0 += NC) {
        int nBlock = imin(NC, N - j0);
        for (int p0 = 0; p0 < K; p0 += KC) {
            int kBlock = imin(KC, K - p0);
            packB(packBufB, B, p0, kBlock, j0, nBlock);
            for (int i0 = 0; i0 < M; i0 += MC) {
                int mBlock = imin(MC, M - i0);
                packA(packBufA, A, i0, mBlock, p0, kBlock);
                for (int jr = 0; jr < nBlock; jr += NR) {
                    Entry *Bpanel = packBufB + (size_t)(jr / NR) * kBlock * NR;
                    for (int ir = 0; ir < mBlock; ir += MR) {
                        Entry *Apanel = packBufA + (size_t)(ir / MR) * kBlock * MR;
                        Matrix Ctile = { C.data + (size_t)(i0 + ir) * C.sRow + (size_t)(j0 + jr) * C.sCol, 0, 0, C.sRow, C.sCol };
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
    if (A.cols <= SMALL_K && C.cols >= NR && C.cols <= SMALL_N && C.sCol == 1)
        smallGemm(A, B, C);
    else
        packedGemm(A, B, C);
}
