// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include "../Operation.h"
#include "../Vector.h"
#include "../Matrix.h"

static void OperationMultiplyForward(Operation *op);
static void OperationMultiplyBackward(Operation *op);
static void multiplyForward(void *arg, int start, int end);
static void multiplyBackwardDa(void *arg, int start, int end);
static void multiplyBackwardDb(void *arg, int start, int end);

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

    // If B is shared and A's batches are contiguous, fuse into a single GEMM
    if (batchStrideB == 0 && batchStrideA == m * strideA_row) {
        memset(op->out.entries + (size_t)start * n, 0, (size_t)(end - start) * n * sizeof(Entry));
        MatrixMultiply((Matrix){op->A.entries + (size_t)start * strideA_row, end - start, k, strideA_row, strideA_col},
                       (Matrix){op->B.entries, k, n, strideB_row, strideB_col},
                       (Matrix){op->out.entries + (size_t)start * n, end - start, n, n, 1});
        return;
    }

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
        MatrixMultiply((Matrix){A_batch + (size_t)r0 * strideA_row, r1 - r0, k, strideA_row, strideA_col},
                       (Matrix){B_batch, k, n, strideB_row, strideB_col},
                       (Matrix){C_batch + (size_t)r0 * n, r1 - r0, n, n, 1});
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
        // If B is shared and dA's batches are contiguous, fuse into a single GEMM
        if (batchStrideB == 0 && batchStrideA == m * aRowStride) {
            MatrixMultiply((Matrix){op->out.gradients + (size_t)start * n, end - start, n, n, 1},
                           (Matrix){op->B.entries, n, k, bColStride, bRowStride},
                           (Matrix){op->A.gradients + (size_t)start * aRowStride, end - start, k, aRowStride, aColStride});
            return;
        }
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
            MatrixMultiply((Matrix){dout + (size_t)r0 * n, r1 - r0, n, n, 1},
                           (Matrix){B, n, k, bColStride, bRowStride},
                           (Matrix){dA + (size_t)r0 * aRowStride, r1 - r0, k, aRowStride, aColStride});
            i = chunkEnd;
        }
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            Entry *B    = op->B.entries     + (size_t)batch * batchStrideB;
            MatrixMultiply((Matrix){B + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride},
                           (Matrix){dout, n, m, 1, n},
                           (Matrix){op->A.gradients + (size_t)start * aColStride, end - start, m, aColStride, aRowStride});
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
            MatrixMultiply((Matrix){A + (size_t)r0 * aColStride, r1 - r0, m, aColStride, aRowStride},
                           (Matrix){dout, m, n, n, 1},
                           (Matrix){dB + (size_t)r0 * bRowStride, r1 - r0, n, bRowStride, bColStride});
            i = chunkEnd;
        }
    } else if (batchStrideA == m * aRowStride) {
        // If A's batches are contiguous, fuse into a single GEMM
        MatrixMultiply((Matrix){op->A.entries + (size_t)start * aColStride, end - start, batchSize * m, aColStride, aRowStride},
                       (Matrix){op->out.gradients, batchSize * m, n, n, 1},
                       (Matrix){op->B.gradients + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride});
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *A    = op->A.entries     + (size_t)batch * batchStrideA;
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            MatrixMultiply((Matrix){A + (size_t)start * aColStride, end - start, m, aColStride, aRowStride},
                           (Matrix){dout, m, n, n, 1},
                           (Matrix){op->B.gradients + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride});
        }
    }
}
