// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include "../Operation.h"
#include "../Vector.h"

typedef struct Matrix { Entry *data; int sRow, sCol; } Matrix;

static int imin(int a, int b) { return a < b ? a : b; }

// An MR x NRV block of vector registers, used by microkernel
#define NRV 2
#define NR  (NRV * VECTOR_DIMS)
#define MR  ((VECTOR_CARD - NRV - 1) / NRV)

// Cache block sizes for the temporary packed A/B blocks used by GEMM
#define KC 256
#define NC (NR * 16)
#define MC (MR * 16)

static void OperationMultiplyForward(Operation *op);
static void OperationMultiplyBackward(Operation *op);
static void multiplyForward(void *arg, int start, int end);
static void multiplyBackwardDa(void *arg, int start, int end);
static void multiplyBackwardDb(void *arg, int start, int end);

static void packB(Entry *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock);
static void packA(Entry *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock);
static void microKernel(Entry *Ap, Entry *Bp, int kBlock, Matrix C, int mValid, int nValid);
void gemm(int N, int K, Matrix A, Matrix B, Matrix C, int r0, int r1);

// Pack a B block into NR-wide panels, padding the right edge with zeros
static void packB(Entry *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock)
{
    int panels = (nBlock + NR - 1) / NR;
    for (int jp = 0; jp < panels; jp++) {
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

// cache-blocked packed GEMM
void gemm(int N, int K, Matrix A, Matrix B, Matrix C, int r0, int r1)
{
    Entry packBufA[MC * KC];
    Entry packBufB[NC * KC];

    for (int j0 = 0; j0 < N; j0 += NC) {
        int nBlock = imin(NC, N - j0);
        for (int p0 = 0; p0 < K; p0 += KC) {
            int kBlock = imin(KC, K - p0);
            packB(packBufB, B, p0, kBlock, j0, nBlock);
            for (int i0 = r0; i0 < r1; i0 += MC) {
                int mBlock = imin(MC, r1 - i0);
                packA(packBufA, A, i0, mBlock, p0, kBlock);
                for (int jr = 0; jr < nBlock; jr += NR) {
                    Entry *Bpanel = packBufB + (size_t)(jr / NR) * kBlock * NR;
                    for (int ir = 0; ir < mBlock; ir += MR) {
                        Entry *Apanel = packBufA + (size_t)(ir / MR) * kBlock * MR;
                        Matrix Ctile = { C.data + (size_t)(i0 + ir) * C.sRow + (size_t)(j0 + jr) * C.sCol, C.sRow, C.sCol };
                        microKernel(Apanel, Bpanel, kBlock, Ctile, imin(MR, mBlock - ir), imin(NR, nBlock - jr));
                    }
                }
            }
        }
    }
}

Tensor OperationMultiply(Tensor A, Tensor B)
{
    int dimsA   = A.dimensions;
    int dimsB   = B.dimensions;
    int dimsOut = dimsA >= dimsB ? dimsA : dimsB;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < dimsOut - 2; i++)
        outShape[i] = (dimsA >= dimsB) ? A.shape[i] : B.shape[i];
    outShape[dimsOut - 2] = A.shape[dimsA - 2];
    outShape[dimsOut - 1] = B.shape[dimsB - 1];
    int autograd = A.gradients != NULL || B.gradients != NULL;
    Tensor out = TensorNew(dimsOut, outShape, autograd);
    return OperationExecute(OperationNew(out, A, B, OperationMultiplyForward, OperationMultiplyBackward, NULL));
}

static void OperationMultiplyForward(Operation *op)
{
    int m         = op->A.shape[op->A.dimensions - 2];
    int k         = op->A.shape[op->A.dimensions - 1];
    int n         = op->B.shape[op->B.dimensions - 1];
    int batchSize = TensorSize(op->out) / (m * n);
    OperationDispatch(multiplyForward, op, batchSize * m, (long)TensorSize(op->out) * k);
}

static void OperationMultiplyBackward(Operation *op)
{
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int batchSize    = TensorSize(op->out) / (m * n);
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;
    long work        = (long)TensorSize(op->out) * k;
    if (op->A.gradients)
        OperationDispatch(multiplyBackwardDa, op, batchStrideA ? batchSize * m : k, work);
    if (op->B.gradients)
        OperationDispatch(multiplyBackwardDb, op, batchStrideB ? batchSize * k : k, work);
}

// out = A @ B
static void multiplyForward(void *arg, int start, int end)
{
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int strideA_row  = op->A.strides[op->A.dimensions - 2];
    int strideA_col  = op->A.strides[op->A.dimensions - 1];
    int strideB_row  = op->B.strides[op->B.dimensions - 2];
    int strideB_col  = op->B.strides[op->B.dimensions - 1];
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;

    int i = start;
    while (i < end) {
        int batch      = i / m;
        int r0         = i % m;
        int batchEnd   = (batch + 1) * m;
        int chunkEnd   = (end < batchEnd) ? end : batchEnd;
        int r1         = r0 + (chunkEnd - i);
        Entry *A_batch = op->A.entries   + (size_t)batch * batchStrideA;
        Entry *B_batch = op->B.entries   + (size_t)batch * batchStrideB;
        Entry *C_batch = op->out.entries + (size_t)batch * m * n;

        memset(C_batch + (size_t)r0 * n, 0, (size_t)(r1 - r0) * n * sizeof(Entry));
        gemm(n, k, (Matrix){A_batch, strideA_row, strideA_col},
                   (Matrix){B_batch, strideB_row, strideB_col},
                   (Matrix){C_batch, n, 1}, r0, r1);
        i = chunkEnd;
    }
}

// dA += dout @ B^T.
static void multiplyBackwardDa(void *arg, int start, int end)
{
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int batchSize    = TensorSize(op->out) / (m * n);
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;
    int aRowStride   = op->A.strides[op->A.dimensions - 2];
    int aColStride   = op->A.strides[op->A.dimensions - 1];
    int bRowStride   = op->B.strides[op->B.dimensions - 2];
    int bColStride   = op->B.strides[op->B.dimensions - 1];

    if (batchStrideA) {
        int i = start;
        while (i < end) {
            int batch    = i / m;
            int r0       = i % m;
            int batchEnd = (batch + 1) * m;
            int chunkEnd = (end < batchEnd) ? end : batchEnd;
            int r1       = r0 + (chunkEnd - i);
            Entry *dout  = op->out.gradients + (size_t)batch * m * n;
            Entry *B     = op->B.entries     + (size_t)batch * batchStrideB;
            Entry *dA    = op->A.gradients   + (size_t)batch * batchStrideA;
            gemm(k, n, (Matrix){dout, n, 1}, (Matrix){B, bColStride, bRowStride},
                       (Matrix){dA, aRowStride, aColStride}, r0, r1);
            i = chunkEnd;
        }
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            Entry *B    = op->B.entries     + (size_t)batch * batchStrideB;
            gemm(m, n, (Matrix){B, bRowStride, bColStride}, (Matrix){dout, 1, n},
                       (Matrix){op->A.gradients, aColStride, aRowStride}, start, end);
        }
    }
}

// dB += A^T @ dout.
static void multiplyBackwardDb(void *arg, int start, int end)
{
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int batchSize    = TensorSize(op->out) / (m * n);
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;
    int aRowStride   = op->A.strides[op->A.dimensions - 2];
    int aColStride   = op->A.strides[op->A.dimensions - 1];
    int bRowStride   = op->B.strides[op->B.dimensions - 2];
    int bColStride   = op->B.strides[op->B.dimensions - 1];

    if (batchStrideB) {
        int i = start;
        while (i < end) {
            int batch    = i / k;
            int r0       = i % k;
            int batchEnd = (batch + 1) * k;
            int chunkEnd = (end < batchEnd) ? end : batchEnd;
            int r1       = r0 + (chunkEnd - i);
            Entry *A     = op->A.entries     + (size_t)batch * batchStrideA;
            Entry *dout  = op->out.gradients + (size_t)batch * m * n;
            Entry *dB    = op->B.gradients   + (size_t)batch * batchStrideB;
            gemm(n, m, (Matrix){A, aColStride, aRowStride}, (Matrix){dout, n, 1},
                       (Matrix){dB, bRowStride, bColStride}, r0, r1);
            i = chunkEnd;
        }
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *A    = op->A.entries     + (size_t)batch * batchStrideA;
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            gemm(n, m, (Matrix){A, aColStride, aRowStride}, (Matrix){dout, n, 1},
                       (Matrix){op->B.gradients, bRowStride, bColStride}, start, end);
        }
    }
}
