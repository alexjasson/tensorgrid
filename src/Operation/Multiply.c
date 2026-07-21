// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include "../Operation.h"
#include "../math/Matrix.h"

static void OperationMultiplyForward(Operation *op);
static void OperationMultiplyBackward(Operation *op);
static void multiplyForward(void *arg, int start, int end);
static void multiplyBackwardDa(void *arg, int start, int end);
static void multiplyBackwardDb(void *arg, int start, int end);

Tensor OperationMultiply(Tensor A, Tensor B)
{
    int dimsA   = TensorDimensions(A);
    int dimsB   = TensorDimensions(B);
    int dimsOut = dimsA >= dimsB ? dimsA : dimsB;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < dimsOut - 2; i++)
        outShape[i] = (dimsA >= dimsB) ? TensorShape(A)[i] : TensorShape(B)[i];
    outShape[dimsOut - 2] = TensorShape(A)[dimsA - 2];
    outShape[dimsOut - 1] = TensorShape(B)[dimsB - 1];
    int autograd = TensorGradients(A) != NULL || TensorGradients(B) != NULL;
    Tensor out = TensorNew(dimsOut, outShape, REAL, autograd);
    return OperationExecute(OperationNew(out, A, B, OperationMultiplyForward, OperationMultiplyBackward, NULL));
}

static void OperationMultiplyForward(Operation *op)
{
    Tensor A = op->A, B = op->B;
    int dimsA = TensorDimensions(A), dimsB = TensorDimensions(B);
    if (TensorDataType(A) != REAL || TensorScale(A)) {
        op->A = TensorCopy(A);
        TensorToType(op->A, REAL);
    }
    int scaledIncompatible = TensorScale(B) && !(dimsB == 2 && TensorStride(B)[dimsB - 1] == 1);
    int integerBatched     = dimsB >= 3 && TensorDataType(B) != REAL;
    if (scaledIncompatible || integerBatched) {
        op->B = TensorCopy(B);
        TensorToType(op->B, REAL);
    }

    int m = TensorShape(op->A)[dimsA - 2], k = TensorShape(op->A)[dimsA - 1], n = TensorShape(op->B)[dimsB - 1];
    int batchSize = TensorSize(op->out) / (m * n);
    OperationDispatch(multiplyForward, op, batchSize * m, (long)TensorSize(op->out) * k);

    if (op->A != A) { TensorFree(op->A); op->A = A; }
    if (op->B != B) { TensorFree(op->B); op->B = B; }
}

static void OperationMultiplyBackward(Operation *op)
{
    int dimsA = TensorDimensions(op->A), dimsB = TensorDimensions(op->B);
    int m = TensorShape(op->A)[dimsA - 2], k = TensorShape(op->A)[dimsA - 1], n = TensorShape(op->B)[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? TensorStride(op->A)[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? TensorStride(op->B)[dimsB - 3] : 0;
    int batchSize = TensorSize(op->out) / (m * n);
    long work     = (long)TensorSize(op->out) * k;

    if (TensorGradients(op->A)) {
        int colSplit = batchStrideA == 0 && (batchSize > 1 || m < k);
        OperationDispatch(multiplyBackwardDa, op, colSplit ? k : batchSize * m, work);
    }
    if (TensorGradients(op->B)) {
        int aRowStride = TensorStride(op->A)[dimsA - 2];
        int reduce = batchStrideB == 0 && batchStrideA == m * aRowStride && batchSize > 1;
        OperationDispatch(multiplyBackwardDb, op, reduce ? batchSize * m : (batchStrideB ? batchSize * k : k), work);
    }
}

// out = A @ B
static void multiplyForward(void *arg, int start, int end)
{
    Operation *op = arg;
    int dimsA = TensorDimensions(op->A), dimsB = TensorDimensions(op->B);
    int m = TensorShape(op->A)[dimsA - 2], k = TensorShape(op->A)[dimsA - 1], n = TensorShape(op->B)[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? TensorStride(op->A)[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? TensorStride(op->B)[dimsB - 3] : 0;
    int aRowStride = TensorStride(op->A)[dimsA - 2], aColStride = TensorStride(op->A)[dimsA - 1];
    int bRowStride = TensorStride(op->B)[dimsB - 2], bColStride = TensorStride(op->B)[dimsB - 1];

    Matrix Bv  = MatrixView(TensorEntries(op->B), k, n, bRowStride, bColStride);
    Bv.type    = TensorDataType(op->B);
    Bv.scales  = TensorScale(op->B);

    // If B is shared and A's batches are contiguous, fuse into a single GEMM
    if (batchStrideB == 0 && batchStrideA == m * aRowStride) {
        memset((Real *)TensorEntries(op->out) + (size_t)start * n, 0, (size_t)(end - start) * n * sizeof(Real));
        MatrixMultiply(MatrixView((Real *)(Real *)TensorEntries(op->A) + (size_t)start * aRowStride, end - start, k, aRowStride, aColStride),
                       Bv,
                       MatrixView((Real *)TensorEntries(op->out) + (size_t)start * n, end - start, n, n, 1));
        return;
    }

    int i = start;
    while (i < end) {
        int batch      = i / m;
        int r0         = i % m;
        int batchEnd   = (batch + 1) * m;
        int chunkEnd   = (end < batchEnd) ? end : batchEnd;
        int r1         = r0 + (chunkEnd - i);
        Real *A_batch = (Real *)(Real *)TensorEntries(op->A) + (size_t)batch * batchStrideA;
        Real *C_batch = (Real *)TensorEntries(op->out) + (size_t)batch * m * n;

        Bv.data = (Real *)TensorEntries(op->B) + (size_t)batch * batchStrideB;
        memset(C_batch + (size_t)r0 * n, 0, (size_t)(r1 - r0) * n * sizeof(Real));
        MatrixMultiply(MatrixView(A_batch + (size_t)r0 * aRowStride, r1 - r0, k, aRowStride, aColStride),
                       Bv,
                       MatrixView(C_batch + (size_t)r0 * n, r1 - r0, n, n, 1));
        i = chunkEnd;
    }
}

// dA = dout @ B^T.
static void multiplyBackwardDa(void *arg, int start, int end)
{
    Operation *op = arg;
    int dimsA = TensorDimensions(op->A), dimsB = TensorDimensions(op->B);
    int m = TensorShape(op->A)[dimsA - 2], k = TensorShape(op->A)[dimsA - 1], n = TensorShape(op->B)[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? TensorStride(op->A)[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? TensorStride(op->B)[dimsB - 3] : 0;
    int aRowStride = TensorStride(op->A)[dimsA - 2], aColStride = TensorStride(op->A)[dimsA - 1];
    int bRowStride = TensorStride(op->B)[dimsB - 2], bColStride = TensorStride(op->B)[dimsB - 1];
    int batchSize = TensorSize(op->out) / (m * n);

    // Column split
    if (batchStrideA == 0 && (batchSize > 1 || m < k)) {
        for (int batch = 0; batch < batchSize; batch++) {
            Real *dout = TensorGradients(op->out) + (size_t)batch * m * n;
            Real *B    = (Real *)TensorEntries(op->B) + (size_t)batch * batchStrideB;
            MatrixMultiply(MatrixView(B + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride),
                           MatrixView(dout, n, m, 1, n),
                           MatrixView(TensorGradients(op->A) + (size_t)start * aColStride, end - start, m, aColStride, aRowStride));
        }
        return;
    }

    // Row split
    if (batchStrideB == 0 && (TensorDimensions(op->A) < 3 || batchStrideA == m * aRowStride)) {
        MatrixMultiply(MatrixView(TensorGradients(op->out) + (size_t)start * n, end - start, n, n, 1),
                       MatrixView(TensorEntries(op->B), n, k, bColStride, bRowStride),
                       MatrixView(TensorGradients(op->A) + (size_t)start * aRowStride, end - start, k, aRowStride, aColStride));
        return;
    }
    int i = start;
    while (i < end) {
        int batch    = i / m;
        int r0       = i % m;
        int batchEnd = (batch + 1) * m;
        int chunkEnd = (end < batchEnd) ? end : batchEnd;
        int r1       = r0 + (chunkEnd - i);
        Real *dout  = TensorGradients(op->out) + (size_t)batch * m * n;
        Real *B     = (Real *)TensorEntries(op->B) + (size_t)batch * batchStrideB;
        Real *dA    = TensorGradients(op->A)   + (size_t)batch * batchStrideA;
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
    int dimsA = TensorDimensions(op->A), dimsB = TensorDimensions(op->B);
    int m = TensorShape(op->A)[dimsA - 2], k = TensorShape(op->A)[dimsA - 1], n = TensorShape(op->B)[dimsB - 1];
    int batchStrideA = dimsA >= 3 ? TensorStride(op->A)[dimsA - 3] : 0;
    int batchStrideB = dimsB >= 3 ? TensorStride(op->B)[dimsB - 3] : 0;
    int aRowStride = TensorStride(op->A)[dimsA - 2], aColStride = TensorStride(op->A)[dimsA - 1];
    int bRowStride = TensorStride(op->B)[dimsB - 2], bColStride = TensorStride(op->B)[dimsB - 1];
    int batchSize = TensorSize(op->out) / (m * n);

    if (batchStrideB == 0 && batchStrideA == m * aRowStride && batchSize > 1) {
        Real local[(size_t)k * n];
        memset(local, 0, (size_t)k * n * sizeof(Real));
        MatrixMultiply(MatrixView((Real *)(Real *)TensorEntries(op->A) + (size_t)start * aRowStride, k, end - start, aColStride, aRowStride),
                       MatrixView(TensorGradients(op->out) + (size_t)start * n, end - start, n, n, 1),
                       MatrixView(local, k, n, n, 1));
        ThreadPool pool = ThreadPoolGet();
        ThreadPoolLock(pool);
        for (int i = 0; i < k; i++)
            for (int j = 0; j < n; j++)
                TensorGradients(op->B)[(size_t)i * bRowStride + (size_t)j * bColStride] += local[(size_t)i * n + j];
        ThreadPoolUnlock(pool);
        return;
    }

    if (batchStrideB) {
        int i = start;
        while (i < end) {
            int batch    = i / k;
            int r0       = i % k;
            int batchEnd = (batch + 1) * k;
            int chunkEnd = (end < batchEnd) ? end : batchEnd;
            int r1       = r0 + (chunkEnd - i);
            Real *A     = (Real *)(Real *)TensorEntries(op->A) + (size_t)batch * batchStrideA;
            Real *dout  = TensorGradients(op->out) + (size_t)batch * m * n;
            Real *dB    = TensorGradients(op->B)   + (size_t)batch * batchStrideB;
            MatrixMultiply(MatrixView(A + (size_t)r0 * aColStride, r1 - r0, m, aColStride, aRowStride),
                           MatrixView(dout, m, n, n, 1),
                           MatrixView(dB + (size_t)r0 * bRowStride, r1 - r0, n, bRowStride, bColStride));
            i = chunkEnd;
        }
    } else if (batchStrideA == m * aRowStride) {
        // If A's batches are contiguous, fuse into a single GEMM
        MatrixMultiply(MatrixView((Real *)(Real *)TensorEntries(op->A) + (size_t)start * aColStride, end - start, batchSize * m, aColStride, aRowStride),
                       MatrixView(TensorGradients(op->out), batchSize * m, n, n, 1),
                       MatrixView(TensorGradients(op->B) + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride));
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Real *A    = (Real *)(Real *)TensorEntries(op->A) + (size_t)batch * batchStrideA;
            Real *dout = TensorGradients(op->out) + (size_t)batch * m * n;
            MatrixMultiply(MatrixView(A + (size_t)start * aColStride, end - start, m, aColStride, aRowStride),
                           MatrixView(dout, m, n, n, 1),
                           MatrixView(TensorGradients(op->B) + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride));
        }
    }
}
