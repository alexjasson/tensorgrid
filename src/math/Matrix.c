// https://github.com/alexjasson/tensorgrid

#include "Matrix.h"
#include "Scale.h"
#include <stddef.h>
#include <stdint.h>
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

static void groupSteps(Matrix M, int *perRow, int *perCol);
static int scaleGroup(int row, int col, int perRow, int perCol);
static int columnGroups(Matrix M);
static void packB(void *buf, Matrix A, Matrix B, int p0, int kBlock, int j0, int nBlock);
static void packA(void *buf, Matrix A, Matrix B, int i0, int mBlock, int p0, int kBlock);
static void microKernelContiguous(const void *Ap, const void *Bp, int kBlock, Matrix C);
static void microKernel(void *Ap, void *Bp, Real *bScale, int kBlock, Matrix C, int mValid, int nValid, int integer);
static void smallGemm(Matrix A, Matrix B, Matrix C);
static void packedGemm(Matrix A, Matrix B, Matrix C);

Matrix MatrixView(Real *data, int rows, int cols, int sRow, int sCol)
{
    return (Matrix){data, rows, cols, sRow, sCol, FLOAT32, 1.0, {NULL, NULL, 0, 0}};
}

static int imin(int a, int b) { return a < b ? a : b; }

static int integerPackB(Matrix A, Matrix B)
{
    if (B.type != INT8 && B.type != INT16 && B.type != UINT8) return 0;
    if (sizeof(Real) != 4) return 0;
    return !columnGroups(B) && !columnGroups(A);
}

static int integerPackA(Matrix A, Matrix B)
{
    if (A.type != INT8 && A.type != INT16 && A.type != UINT8) return 0;
    if (B.type != FLOAT32 || sizeof(Real) != 4) return 0;
    if (columnGroups(A)) return 0;
    return (long)A.rows * A.cols >= 1024L * B.cols;
}

static void groupSteps(Matrix M, int *perRow, int *perCol)
{
    *perRow = *perCol = 0;
    if (!M.scale.values || !M.scale.stride) return;
    long down = (long)(M.rows - 1) * M.sRow, across = (long)(M.cols - 1) * M.sCol;
    if (down + across < M.scale.stride) return;

    int *step = across < M.scale.stride ? perRow : perCol;
    int span  = across < M.scale.stride ? M.sRow : M.sCol;
    *step = span > M.scale.stride ? span / M.scale.stride : 1;
}

static int scaleGroup(int row, int col, int perRow, int perCol)
{
    return row * perRow + col * perCol;
}

static int columnGroups(Matrix M)
{
    int perRow, perCol;
    groupSteps(M, &perRow, &perCol);
    return perCol != 0;
}

static int integerAt(Matrix M, size_t e, int group)
{
    int v = M.type == INT8  ? ((const int8_t  *)M.data)[e]
          : M.type == UINT8 ? ((const uint8_t *)M.data)[e]
                            : ((const int16_t *)M.data)[e];
    return v - (int)ScaleZero(M.scale, group);
}

static Real realAt(Matrix M, const unsigned char *e, int group)
{
    Real v = M.type == FLOAT32 ? *(const Real *)e : DataGet(e, M.type);
    return v - ScaleZero(M.scale, group);
}

// Pack a B block into NR-wide panels, padding the right edge with zeros
static void packB(void *buf, Matrix A, Matrix B, int p0, int kBlock, int j0, int nBlock)
{
    int panels = (nBlock + NR - 1) / NR;
    int rowB, colB;
    groupSteps(B, &rowB, &colB);
    int fullPanels = (B.sCol == 1) ? nBlock / NR : 0;

    // Integer path (quantized A)
    if (integerPackA(A, B)) {
        int kPairs = (kBlock + 1) / 2;
        size_t panelSize = NR * sizeof(Real) + (size_t)kPairs * 2 * NR * sizeof(int16_t);
        Real qMax = imin(32767, INT32_MAX / (kBlock * (A.type == INT8 ? 127 : A.type == UINT8 ? 255 : 32767)));

        for (int jp = 0; jp < panels; jp++) {
            Real *colScale = (Real *)((char *)buf + jp * panelSize);
            int16_t *dst = (int16_t *)(colScale + NR);
            Real vals[KC][NR];
            Real m[NR] = {0};
            for (int p = 0; p < kBlock; p++) {
                Real s = B.factor * ScaleValue(B.scale, scaleGroup(p0 + p, j0, rowB, colB));
                for (int nr = 0; nr < NR; nr++) {
                    int col = jp * NR + nr;
                    Real f = col < nBlock ? s * B.data[(size_t)(p0 + p) * B.sRow + (size_t)(j0 + col) * B.sCol] : 0;
                    vals[p][nr] = f;
                    m[nr] = f < -m[nr] ? -f : f > m[nr] ? f : m[nr];
                }
            }

            Real inv[NR];
            for (int nr = 0; nr < NR; nr++) {
                inv[nr] = m[nr] > 0 ? qMax / m[nr] : 0;
                colScale[nr] = m[nr] / qMax;
            }

            for (int p2 = 0; p2 < kPairs; p2++)
                for (int nr = 0; nr < NR; nr++)
                    for (int i = 0; i < 2; i++) {
                        int p = 2 * p2 + i;
                        Real f = p < kBlock ? vals[p][nr] * inv[nr] : 0;
                        dst[(p2 * NR + nr) * 2 + i] = (int16_t)(f < 0 ? f - 0.5f : f + 0.5f);
                    }
        }
        return;
    }

    // Integer path (quantized B)
    if (integerPackB(A, B)) {
        int kPairs = (kBlock + 1) / 2;

        for (int jp = 0; jp < fullPanels; jp++) {
            int16_t *dst = (int16_t *)buf + (size_t)jp * kPairs * 2 * NR;
            size_t src = (size_t)p0 * B.sRow + j0 + (size_t)jp * NR;
            for (int p2 = 0; p2 < kBlock / 2; p2++) {
                int g0 = scaleGroup(p0 + 2 * p2, j0, rowB, colB), g1 = scaleGroup(p0 + 2 * p2 + 1, j0, rowB, colB);
                for (int nr = 0; nr < NR; nr++) {
                    size_t e = src + (size_t)(2 * p2) * B.sRow + nr;
                    dst[(p2 * NR + nr) * 2]     = integerAt(B, e, g0);
                    dst[(p2 * NR + nr) * 2 + 1] = integerAt(B, e + B.sRow, g1);
                }
            }
            if (kBlock % 2) {
                int g = scaleGroup(p0 + kBlock - 1, j0, rowB, colB);
                for (int nr = 0; nr < NR; nr++) {
                    size_t e = src + (size_t)(kBlock - 1) * B.sRow + nr;
                    dst[((kBlock / 2) * NR + nr) * 2]     = integerAt(B, e, g);
                    dst[((kBlock / 2) * NR + nr) * 2 + 1] = 0;
                }
            }
        }

        for (int jp = fullPanels; jp < panels; jp++) {
            int16_t *dst = (int16_t *)buf + (size_t)jp * kPairs * 2 * NR;
            for (int p2 = 0; p2 < kPairs; p2++)
                for (int i = 0; i < 2; i++) {
                    int p = 2 * p2 + i;
                    int g = p < kBlock ? scaleGroup(p0 + p, j0, rowB, colB) : 0;
                    for (int nr = 0; nr < NR; nr++) {
                        int col = jp * NR + nr;
                        size_t e = (size_t)(p0 + p) * B.sRow + (size_t)(j0 + col) * B.sCol;
                        dst[(p2 * NR + nr) * 2 + i] = p < kBlock && col < nBlock ? integerAt(B, e, g) : 0;
                    }
                }
        }
        return;
    }

    // Default path
    Real *Bp = buf;
    if (B.type == FLOAT32 && B.factor == 1.0 && !B.scale.values) {
        // Vectorize contiguous rows
        for (int jp = 0; jp < fullPanels; jp++) {
            Real *dst = Bp + (size_t)jp * kBlock * NR;
            Real *src = B.data + (size_t)p0 * B.sRow + j0 + (size_t)jp * NR;
            for (int p = 0; p < kBlock; p++)
                for (int v = 0; v < NR; v += VECTOR_DIMS)
                    VectorStore(dst + p * NR + v, VectorLoad(src + (size_t)p * B.sRow + v, FLOAT32));
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

    size_t esz = DataSize(B.type);
    const unsigned char *base = (const unsigned char *)B.data;

    int byCol = colB != 0 && rowB == 0;

    for (int jp = 0; jp < fullPanels; jp++) {
        Real *dst = Bp + (size_t)jp * kBlock * NR;
        const unsigned char *src = base + ((size_t)p0 * B.sRow + j0 + (size_t)jp * NR) * esz;
        Real cs[NR], cz[NR];
        if (byCol)
            for (int nr = 0; nr < NR; nr++) {
                int group = scaleGroup(0, j0 + jp * NR + nr, rowB, colB);
                cs[nr] = B.factor * ScaleValue(B.scale, group);
                cz[nr] = ScaleZero(B.scale, group);
            }
        for (int p = 0; p < kBlock; p++) {
            int group = scaleGroup(p0 + p, j0 + jp * NR, rowB, colB);
            for (int v = 0; v < NR; v += VECTOR_DIMS) {
                Vector scale = byCol ? VectorLoad(cs + v, FLOAT32) : VectorBcast(B.factor * ScaleValue(B.scale, group));
                Vector zero  = byCol ? VectorLoad(cz + v, FLOAT32) : VectorBcast(ScaleZero(B.scale, group));
                VectorStore(dst + p * NR + v,
                            VectorMul(VectorSub(VectorLoad(src + ((size_t)p * B.sRow + v) * esz, B.type), zero), scale));
            }
        }
    }

    for (int jp = fullPanels; jp < panels; jp++) {
        Real *dst = Bp + (size_t)jp * kBlock * NR;
        for (int p = 0; p < kBlock; p++) {
            const unsigned char *brow = base + (size_t)(p0 + p) * B.sRow * esz;
            for (int nr = 0; nr < NR; nr++) {
                int col = jp * NR + nr;
                int group = scaleGroup(p0 + p, j0 + col, rowB, colB);
                dst[p * NR + nr] = (col < nBlock)
                    ? B.factor * ScaleValue(B.scale, group) * realAt(B, brow + (size_t)(j0 + col) * B.sCol * esz, group)
                    : 0.0f;
            }
        }
    }
}

// Pack an A block into MR-row panels, padding the bottom edge with zeros
static void packA(void *buf, Matrix A, Matrix B, int i0, int mBlock, int p0, int kBlock)
{
    int panels = (mBlock + MR - 1) / MR;
    int rowA, colA, rowB, colB;
    groupSteps(A, &rowA, &colA);
    groupSteps(B, &rowB, &colB);
    size_t esz = DataSize(A.type);
    const unsigned char *base = (const unsigned char *)A.data;

    // Integer path (quantized A)
    if (integerPackA(A, B)) {
        int kPairs = (kBlock + 1) / 2;
        size_t panelSize = MR * sizeof(Real) + (size_t)kPairs * 2 * MR * sizeof(int16_t);

        for (int ip = 0; ip < panels; ip++) {
            Real *rowScale = (Real *)((char *)buf + ip * panelSize);
            int16_t *dst = (int16_t *)(rowScale + MR);
            for (int mr = 0; mr < MR; mr++) {
                int row = ip * MR + mr;
                int group = row < mBlock ? scaleGroup(i0 + row, p0, rowA, colA) : 0;
                rowScale[mr] = row < mBlock ? A.factor * ScaleValue(A.scale, group) : 0;
                for (int p2 = 0; p2 < kPairs; p2++)
                    for (int i = 0; i < 2; i++) {
                        int p = 2 * p2 + i;
                        size_t e = (size_t)(i0 + row) * A.sRow + (size_t)(p0 + p) * A.sCol;
                        dst[(p2 * MR + mr) * 2 + i] = row < mBlock && p < kBlock
                            ? integerAt(A, e, group) : 0;
                    }
            }
        }
        return;
    }

    // Integer path (quantized B)
    if (integerPackB(A, B)) {
        int kPairs = (kBlock + 1) / 2;
        size_t panelSize = MR * sizeof(Real) + (size_t)kPairs * 2 * MR * sizeof(int16_t);
        Real qMax = imin(32767, INT32_MAX / (kBlock * (B.type == INT8 ? 127 : B.type == UINT8 ? 255 : 32767)));
        Real sB[KC];
        for (int p = 0; p < kBlock; p++)
            sB[p] = B.factor * ScaleValue(B.scale, scaleGroup(p0 + p, 0, rowB, colB));

        for (int ip = 0; ip < panels; ip++) {
            Real *rowScale = (Real *)((char *)buf + ip * panelSize);
            int16_t *dst = (int16_t *)(rowScale + MR);
            for (int mr = 0; mr < MR; mr++) {
                int row = ip * MR + mr;
                if (row >= mBlock) {
                    for (int p2 = 0; p2 < kPairs; p2++)
                        for (int i = 0; i < 2; i++)
                            dst[(p2 * MR + mr) * 2 + i] = 0;
                    rowScale[mr] = 0;
                    continue;
                }

                const unsigned char *a = base + ((size_t)(i0 + row) * A.sRow + (size_t)p0 * A.sCol) * esz;
                int group = scaleGroup(i0 + row, p0, rowA, colA);
                Real vals[KC];
                Real m = 0;
                for (int p = 0; p < kBlock; p++) {
                    const unsigned char *e = a + (size_t)p * A.sCol * esz;
                    Real f = realAt(A, e, group) * sB[p];
                    vals[p] = f;
                    m = f < -m ? -f : f > m ? f : m;
                }
                Real inv = m > 0 ? qMax / m : 0;
                rowScale[mr] = A.factor * ScaleValue(A.scale, group) * m / qMax;

                for (int p2 = 0; p2 < kPairs; p2++)
                    for (int i = 0; i < 2; i++) {
                        int p = 2 * p2 + i;
                        Real f = p < kBlock ? vals[p] * inv : 0;
                        dst[(p2 * MR + mr) * 2 + i] = (int16_t)(f < 0 ? f - 0.5f : f + 0.5f);
                    }
            }
        }
        return;
    }

    // Default path
    Real *Ap = buf;
    if (A.type == FLOAT32 && !A.scale.values) {
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

    for (int ip = 0; ip < panels; ip++) {
        Real *dst = Ap + (size_t)ip * kBlock * MR;
        for (int p = 0; p < kBlock; p++) {
            for (int mr = 0; mr < MR; mr++) {
                int row = ip * MR + mr;
                int group = scaleGroup(i0 + row, p0 + p, rowA, colA);
                dst[p * MR + mr] = (row < mBlock)
                    ? ScaleValue(A.scale, group) * realAt(A, base + ((size_t)(i0 + row) * A.sRow + (size_t)(p0 + p) * A.sCol) * esz, group)
                    : 0.0f;
            }
        }
    }
}

static void microKernelContiguous(const void *Ap, const void *Bp, int kBlock, Matrix C)
{
    Vector acc[MR][NRV];
    for (int mr = 0; mr < MR; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VectorZero();

    for (int mr = 0; mr < MR; mr++) {
        VectorFetch(C.data + (size_t)mr * C.sRow);
        VectorFetch(C.data + (size_t)mr * C.sRow + NR - 1);
    }

    const Real *aRow = Ap;
    const Real *bRow = Bp;
    for (int p = 0; p < kBlock; p++, aRow += MR, bRow += NR) {
        Vector b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VectorLoad(bRow + v * VECTOR_DIMS, FLOAT32);
        for (int mr = 0; mr < MR; mr++) {
            Vector a = VectorBcast(aRow[mr]);
            for (int v = 0; v < NRV; v++)
                acc[mr][v] = VectorFma(a, b[v], acc[mr][v]);
        }
    }

    for (int mr = 0; mr < MR; mr++) {
        Real *Crow = C.data + (size_t)mr * C.sRow;
        for (int v = 0; v < NRV; v++)
            VectorStore(Crow + v * VECTOR_DIMS, VectorAdd(VectorLoad(Crow + v * VECTOR_DIMS, FLOAT32), acc[mr][v]));
    }
}

static void microKernel(void *Ap, void *Bp, Real *bScale, int kBlock, Matrix C, int mValid, int nValid, int integer)
{
    for (int mr = 0; mr < mValid; mr++) {
        VectorFetch(C.data + (size_t)mr * C.sRow);
        VectorFetch(C.data + (size_t)mr * C.sRow + (size_t)(NR - 1) * C.sCol);
    }

    // Integer path
    if (integer) {
        if (C.sCol == 1 && mValid == MR && nValid == NR) {
            VectorDot(Ap, Bp, C.data, bScale, C.sRow, kBlock);
            return;
        }

        Real tile[MR * NR] = {0};
        VectorDot(Ap, Bp, tile, bScale, NR, kBlock);
        for (int mr = 0; mr < mValid; mr++)
            for (int nr = 0; nr < nValid; nr++)
                C.data[mr * C.sRow + nr * C.sCol] += tile[mr * NR + nr];
        return;
    }

    // Default path
    Vector acc[MR][NRV];
    for (int mr = 0; mr < mValid; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VectorZero();

    const Real *aRow = Ap;
    const Real *bRow = Bp;
    for (int p = 0; p < kBlock; p++, aRow += MR, bRow += NR) {
        Vector b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VectorLoad(bRow + v * VECTOR_DIMS, FLOAT32);
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
                VectorStore(Crow + v * VECTOR_DIMS, VectorAdd(VectorLoad(Crow + v * VECTOR_DIMS, FLOAT32), acc[mr][v]));
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

    B.factor = A.factor * B.factor / C.factor;
    if (B.sCol == 1 && B.type == FLOAT32 && B.factor == 1.0 && !B.scale.values) {
        Bd = B.data;
        sB = B.sRow;
    } else if (B.type == FLOAT32 && B.factor == 1.0 && !B.scale.values) {
        for (int p = 0; p < K; p++)
            for (int j = 0; j < N; j++)
                Bpack[(size_t)p * N + j] = B.data[(size_t)p * B.sRow + (size_t)j * B.sCol];
        Bd = Bpack;
        sB = N;
    } else {
        size_t esz = DataSize(B.type);
        const unsigned char *base = (const unsigned char *)B.data;
        int rowB, colB;
        groupSteps(B, &rowB, &colB);
        for (int p = 0; p < K; p++)
            for (int j = 0; j < N; j++) {
                int group = scaleGroup(p, j, rowB, colB);
                Bpack[(size_t)p * N + j] = B.factor * ScaleValue(B.scale, group)
                                         * realAt(B, base + ((size_t)p * B.sRow + (size_t)j * B.sCol) * esz, group);
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
                    for (int v = 0; v < NRV; v++) b[v] = VectorLoad(Brow + v * VECTOR_DIMS, FLOAT32);
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
                    for (int v = 0; v < NRV; v++) b[v] = VectorLoad(Brow + v * VECTOR_DIMS, FLOAT32);
                    for (int m = 0; m < mValid; m++) {
                        Vector a = VectorBcast(Acol[(size_t)m * sA]);
                        for (int v = 0; v < NRV; v++) acc[m][v] = VectorFma(a, b[v], acc[m][v]);
                    }
                }
            }

            for (int m = 0; m < mValid; m++) {
                Real *Cm = C.data + (size_t)(i0 + m) * sC + j0;
                for (int v = 0; v < NRV; v++)
                    VectorStore(Cm + v * VECTOR_DIMS, VectorAdd(VectorLoad(Cm + v * VECTOR_DIMS, FLOAT32), acc[m][v]));
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
    int integerB = integerPackB(A, B);
    int integerA = integerPackA(A, B);
    int integer  = integerA || integerB;

    union { Real r[MC * KC]; int16_t i[MC * (KC + 2)]; } bufA;
    union { Real r[NC * KC]; int16_t i[NC * (KC + 2)]; } bufB;

    if (integer)
        A.factor /= C.factor;
    else
        B.factor = A.factor * B.factor / C.factor;

    for (int j0 = 0; j0 < N; j0 += NC) {
        int nBlock = imin(NC, N - j0);
        for (int p0 = 0; p0 < K; p0 += KC) {
            int kBlock = imin(KC, K - p0);
            int kPairs = (kBlock + 1) / 2;
            size_t panelA = MR * sizeof(Real) + (size_t)kPairs * 2 * MR * sizeof(int16_t);
            size_t panelB = NR * sizeof(Real) + (size_t)kPairs * 2 * NR * sizeof(int16_t);
            packB(&bufB, A, B, p0, kBlock, j0, nBlock);
            for (int i0 = 0; i0 < M; i0 += MC) {
                int mBlock = imin(MC, M - i0);
                packA(&bufA, A, B, i0, mBlock, p0, kBlock);
                for (int jr = 0; jr < nBlock; jr += NR) {
                    Real *bScale = integerA ? (Real *)((char *)&bufB + (size_t)(jr / NR) * panelB) : NULL;
                    void *Bpanel = integerA ? (void *)(bScale + NR)
                                 : integerB ? (void *)(bufB.i + (size_t)(jr / NR) * kPairs * 2 * NR)
                                            : (void *)(bufB.r + (size_t)(jr / NR) * kBlock * NR);
                    for (int ir = 0; ir < mBlock; ir += MR) {
                        void *Apanel = integer ? (void *)((char *)&bufA + (size_t)(ir / MR) * panelA)
                                               : (void *)(bufA.r + (size_t)(ir / MR) * kBlock * MR);
                        Matrix Ctile = C;
                        Ctile.data = C.data + (size_t)(i0 + ir) * C.sRow + (size_t)(j0 + jr) * C.sCol;
                        int mValid = imin(MR, mBlock - ir);
                        int nValid = imin(NR, nBlock - jr);
                        if (C.sCol == 1 && mValid == MR && nValid == NR && !integer)
                            microKernelContiguous(Apanel, Bpanel, kBlock, Ctile);
                        else
                            microKernel(Apanel, Bpanel, bScale, kBlock, Ctile, mValid, nValid, integer);
                    }
                }
            }
        }
    }
}

void MatrixMultiply(Matrix A, Matrix B, Matrix C)
{
    int integer = integerPackB(A, B);
    if (!integer && A.type == FLOAT32 && !A.scale.values && A.cols <= SMALL_K && C.cols >= NR && C.cols <= SMALL_N && C.sCol == 1)
        smallGemm(A, B, C);
    else
        packedGemm(A, B, C);
}
