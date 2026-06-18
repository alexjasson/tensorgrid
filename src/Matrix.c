// https://github.com/alexjasson/tensorgrid

#include "Matrix.h"
#include "Vector.h"

static int imin(int a, int b) { return a < b ? a : b; }

// An MR x NRV block of vector registers, used by microKernel
#define NRV 2
#define NR  (NRV * VECTOR_DIMS)
#define MR  ((VECTOR_CARD - NRV - 1) / NRV)

// Cache block sizes for the temporary packed A/B blocks
#define KC 256
#define NC (NR * 32)
#define MC (MR * 16)

static void packB(Entry *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock);
static void packA(Entry *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock);
static void microKernel(Entry *Ap, Entry *Bp, int kBlock, Matrix C, int mValid, int nValid);

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

// Compute one packed MR x NR tile and add the valid region back into C
static void microKernel(Entry *Ap, Entry *Bp, int kBlock, Matrix C, int mValid, int nValid)
{
    Vector acc[MR][NRV];
    for (int mr = 0; mr < MR; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VectorZero();

    // Prefetch the C tile
    for (int mr = 0; mr < mValid; mr++) {
        VectorFetch(C.data + (size_t)mr * C.sRow);
        VectorFetch(C.data + (size_t)mr * C.sRow + NR - 1);
    }

    for (int p = 0; p < kBlock; p++) {
        Vector b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VectorLoad(Bp + p * NR + v * VECTOR_DIMS);
        for (int mr = 0; mr < MR; mr++) {
            Vector a = VectorBcast(Ap[p * MR + mr]);
            for (int v = 0; v < NRV; v++)
                acc[mr][v] = VectorFma(a, b[v], acc[mr][v]);
        }
    }

    Entry tile[MR][NR];
    for (int mr = 0; mr < MR; mr++)
        for (int v = 0; v < NRV; v++)
            VectorStore(&tile[mr][v * VECTOR_DIMS], acc[mr][v]);
    for (int mr = 0; mr < mValid; mr++)
        for (int nr = 0; nr < nValid; nr++)
            C.data[mr * C.sRow + nr * C.sCol] += tile[mr][nr];
}

// cache-blocked packed matrix multiply
void MatrixMultiply(Matrix A, Matrix B, Matrix C)
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
                        microKernel(Apanel, Bpanel, kBlock, Ctile, imin(MR, mBlock - ir), imin(NR, nBlock - jr));
                    }
                }
            }
        }
    }
}
