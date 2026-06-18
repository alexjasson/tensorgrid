// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include "../Operation.h"
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
    int dimsA = op->A.dimensions, dimsB = op->B.dimensions;
    int m = op->A.shape[dimsA - 2], k = op->A.shape[dimsA - 1], n = op->B.shape[dimsB - 1];
    int batchSize = TensorSize(op->out) / (m * n);
    OperationDispatch(multiplyForward, op, batchSize * m, (long)TensorSize(op->out) * k);
}

static void OperationMultiplyBackward(Operation *op)
{
    int dimsA = op->A.dimensions, dimsB = op->B.dimensions;
    int m = op->A.shape[dimsA - 2], k = op->A.shape[dimsA - 1], n = op->B.shape[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? op->A.strides[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? op->B.strides[dimsB - 3] : 0;
    int batchSize = TensorSize(op->out) / (m * n);
    long work     = (long)TensorSize(op->out) * k;

    if (op->A.gradients) {
        int colSplit = batchStrideA == 0 && (batchSize > 1 || m < k);
        OperationDispatch(multiplyBackwardDa, op, colSplit ? k : batchSize * m, work);
    }
    if (op->B.gradients)
        OperationDispatch(multiplyBackwardDb, op, batchStrideB ? batchSize * k : k, work);
}

// out = A @ B
static void multiplyForward(void *arg, int start, int end)
{
    Operation *op = arg;
    int dimsA = op->A.dimensions, dimsB = op->B.dimensions;
    int m = op->A.shape[dimsA - 2], k = op->A.shape[dimsA - 1], n = op->B.shape[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? op->A.strides[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? op->B.strides[dimsB - 3] : 0;
    int aRowStride = op->A.strides[dimsA - 2], aColStride = op->A.strides[dimsA - 1];
    int bRowStride = op->B.strides[dimsB - 2], bColStride = op->B.strides[dimsB - 1];

    // If B is shared and A's batches are contiguous, fuse into a single GEMM
    if (batchStrideB == 0 && batchStrideA == m * aRowStride) {
        memset(op->out.entries + (size_t)start * n, 0, (size_t)(end - start) * n * sizeof(Entry));
        MatrixMultiply(MatrixView(op->A.entries + (size_t)start * aRowStride, end - start, k, aRowStride, aColStride),
                       MatrixView(op->B.entries, k, n, bRowStride, bColStride),
                       MatrixView(op->out.entries + (size_t)start * n, end - start, n, n, 1));
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
        MatrixMultiply(MatrixView(A_batch + (size_t)r0 * aRowStride, r1 - r0, k, aRowStride, aColStride),
                       MatrixView(B_batch, k, n, bRowStride, bColStride),
                       MatrixView(C_batch + (size_t)r0 * n, r1 - r0, n, n, 1));
        i = chunkEnd;
    }
}

// dA = dout @ B^T.
static void multiplyBackwardDa(void *arg, int start, int end)
{
    Operation *op = arg;
    int dimsA = op->A.dimensions, dimsB = op->B.dimensions;
    int m = op->A.shape[dimsA - 2], k = op->A.shape[dimsA - 1], n = op->B.shape[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? op->A.strides[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? op->B.strides[dimsB - 3] : 0;
    int aRowStride = op->A.strides[dimsA - 2], aColStride = op->A.strides[dimsA - 1];
    int bRowStride = op->B.strides[dimsB - 2], bColStride = op->B.strides[dimsB - 1];
    int batchSize = TensorSize(op->out) / (m * n);

    // Column split
    if (batchStrideA == 0 && (batchSize > 1 || m < k)) {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            Entry *B    = op->B.entries     + (size_t)batch * batchStrideB;
            MatrixMultiply(MatrixView(B + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride),
                           MatrixView(dout, n, m, 1, n),
                           MatrixView(op->A.gradients + (size_t)start * aColStride, end - start, m, aColStride, aRowStride));
        }
        return;
    }

    // Row split
    if (batchStrideB == 0 && (op->A.dimensions < 3 || batchStrideA == m * aRowStride)) {
        MatrixMultiply(MatrixView(op->out.gradients + (size_t)start * n, end - start, n, n, 1),
                       MatrixView(op->B.entries, n, k, bColStride, bRowStride),
                       MatrixView(op->A.gradients + (size_t)start * aRowStride, end - start, k, aRowStride, aColStride));
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
        MatrixMultiply(MatrixView(dout + (size_t)r0 * n, r1 - r0, n, n, 1),
                       MatrixView(B, n, k, bColStride, bRowStride),
                       MatrixView(dA + (size_t)r0 * aRowStride, r1 - r0, k, aRowStride, aColStride));
        i = chunkEnd;
    }
}

// dB = A^T @ dout.
static void multiplyBackwardDb(void *arg, int start, int end)
{
    Operation *op = arg;
    int dimsA = op->A.dimensions, dimsB = op->B.dimensions;
    int m = op->A.shape[dimsA - 2], k = op->A.shape[dimsA - 1], n = op->B.shape[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? op->A.strides[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? op->B.strides[dimsB - 3] : 0;
    int aRowStride = op->A.strides[dimsA - 2], aColStride = op->A.strides[dimsA - 1];
    int bRowStride = op->B.strides[dimsB - 2], bColStride = op->B.strides[dimsB - 1];
    int batchSize = TensorSize(op->out) / (m * n);

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
            MatrixMultiply(MatrixView(A + (size_t)r0 * aColStride, r1 - r0, m, aColStride, aRowStride),
                           MatrixView(dout, m, n, n, 1),
                           MatrixView(dB + (size_t)r0 * bRowStride, r1 - r0, n, bRowStride, bColStride));
            i = chunkEnd;
        }
    } else if (batchStrideA == m * aRowStride) {
        // If A's batches are contiguous, fuse into a single GEMM
        MatrixMultiply(MatrixView(op->A.entries + (size_t)start * aColStride, end - start, batchSize * m, aColStride, aRowStride),
                       MatrixView(op->out.gradients, batchSize * m, n, n, 1),
                       MatrixView(op->B.gradients + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride));
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *A    = op->A.entries     + (size_t)batch * batchStrideA;
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            MatrixMultiply(MatrixView(A + (size_t)start * aColStride, end - start, m, aColStride, aRowStride),
                           MatrixView(dout, m, n, n, 1),
                           MatrixView(op->B.gradients + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride));
        }
    }
}
