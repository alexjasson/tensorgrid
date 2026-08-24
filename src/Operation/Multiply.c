// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include "../Operation.h"
#include "../math/Matrix.h"
#include "../math/Vector.h"

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
    int outShape[dimsOut];
    for (int i = 0; i < dimsOut - 2; i++)
        outShape[i] = (dimsA >= dimsB) ? TensorShape(A)[i] : TensorShape(B)[i];
    outShape[dimsOut - 2] = TensorShape(A)[dimsA - 2];
    outShape[dimsOut - 1] = TensorShape(B)[dimsB - 1];
    int autograd = TensorGradients(A) != NULL || TensorGradients(B) != NULL;
    Tensor out = TensorNew(dimsOut, outShape, FLOAT32, autograd);
    return OperationExecute(OperationNew(out, A, B, OperationMultiplyForward, OperationMultiplyBackward, NULL));
}

static void OperationMultiplyForward(Operation *op)
{
    int dimsA = TensorDimensions(op->A), dimsB = TensorDimensions(op->B);
    int m = TensorShape(op->A)[dimsA - 2], k = TensorShape(op->A)[dimsA - 1], n = TensorShape(op->B)[dimsB - 1];
    int batchSize = TensorSize(op->out) / (m * n);
    OperationDispatch(multiplyForward, op, batchSize * m, (long)TensorSize(op->out) * k);
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
        int reduce = batchStrideB == 0 && batchStrideA == m * aRowStride && batchSize > 1 && (dimsA < 3 || !ScaleGet(op->A).values);
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
    unsigned char *Ae = TensorEntries(op->A), *Be = TensorEntries(op->B);
    size_t aSize = DataSize(TensorDataType(op->A)), bSize = DataSize(TensorDataType(op->B));
    Scale aScale = ScaleGet(op->A), bScale = ScaleGet(op->B);

    Matrix Av  = MatrixView((Real *)Ae, m, k, aRowStride, aColStride);
    Av.type    = TensorDataType(op->A);
    Av.scale   = aScale;
    Matrix Bv  = MatrixView((Real *)Be, k, n, bRowStride, bColStride);
    Bv.type    = TensorDataType(op->B);
    Bv.scale   = bScale;

    // If B is shared and A's batches are contiguous, fuse into a single GEMM
    if (batchStrideB == 0 && batchStrideA == m * aRowStride && (dimsA < 3 || !aScale.values)) {
        memset((Real *)TensorEntries(op->out) + (size_t)start * n, 0, (size_t)(end - start) * n * sizeof(Real));
        Av.data = (Real *)(Ae + (size_t)start * aRowStride * aSize);
        Av.rows = end - start;
        Av.scale.values = aScale.values ? aScale.values + start : NULL;
        Av.scale.zeros  = aScale.zeros  ? aScale.zeros  + start : NULL;
        MatrixMultiply(Av, Bv, MatrixView((Real *)TensorEntries(op->out) + (size_t)start * n, end - start, n, n, 1));
        return;
    }

    int i = start;
    while (i < end) {
        int batch      = i / m;
        int r0         = i % m;
        int batchEnd   = (batch + 1) * m;
        int chunkEnd   = (end < batchEnd) ? end : batchEnd;
        int r1         = r0 + (chunkEnd - i);
        Real *C_batch = (Real *)TensorEntries(op->out) + (size_t)batch * m * n;
        int aGroup = dimsA >= 3 ? batch : r0, bGroup = dimsB >= 3 ? batch : 0;

        Av.data = (Real *)(Ae + ((size_t)batch * batchStrideA + (size_t)r0 * aRowStride) * aSize);
        Av.rows = r1 - r0;
        Av.scale.values = aScale.values ? aScale.values + aGroup : NULL;
        Av.scale.zeros  = aScale.zeros  ? aScale.zeros  + aGroup : NULL;

        Bv.data = (Real *)(Be + (size_t)batch * batchStrideB * bSize);
        Bv.scale.values = bScale.values ? bScale.values + bGroup : NULL;
        Bv.scale.zeros  = bScale.zeros  ? bScale.zeros  + bGroup : NULL;

        memset(C_batch + (size_t)r0 * n, 0, (size_t)(r1 - r0) * n * sizeof(Real));
        MatrixMultiply(Av,
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
    unsigned char *Be = TensorEntries(op->B);
    size_t bSize = DataSize(TensorDataType(op->B));
    Scale bScale = ScaleGet(op->B);

    // Column split
    if (batchStrideA == 0 && (batchSize > 1 || m < k)) {
        Matrix Bv  = MatrixView((Real *)Be, end - start, n, bRowStride, bColStride);
        Bv.type    = TensorDataType(op->B);
        Bv.scale   = bScale;
        for (int batch = 0; batch < batchSize; batch++) {
            Real *dout = TensorGradients(op->out) + (size_t)batch * m * n;
            int bGroup = dimsB >= 3 ? batch : start;
            Bv.data = (Real *)(Be + ((size_t)batch * batchStrideB + (size_t)start * bRowStride) * bSize);
            Bv.scale.values = bScale.values ? bScale.values + bGroup : NULL;
            Bv.scale.zeros  = bScale.zeros  ? bScale.zeros  + bGroup : NULL;
            MatrixMultiply(Bv,
                           MatrixView(dout, n, m, 1, n),
                           MatrixView(TensorGradients(op->A) + (size_t)start * aColStride, end - start, m, aColStride, aRowStride));
        }
        return;
    }

    Matrix Bt  = MatrixView((Real *)Be, n, k, bColStride, bRowStride);
    Bt.type    = TensorDataType(op->B);
    Bt.scale   = bScale;

    // Row split
    if (batchStrideB == 0 && (dimsA < 3 || batchStrideA == m * aRowStride)) {
        MatrixMultiply(MatrixView(TensorGradients(op->out) + (size_t)start * n, end - start, n, n, 1),
                       Bt,
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
        Real *dA    = TensorGradients(op->A)   + (size_t)batch * batchStrideA;
        int bGroup  = dimsB >= 3 ? batch : 0;

        Bt.data = (Real *)(Be + (size_t)batch * batchStrideB * bSize);
        Bt.scale.values = bScale.values ? bScale.values + bGroup : NULL;
        Bt.scale.zeros  = bScale.zeros  ? bScale.zeros  + bGroup : NULL;
        MatrixMultiply(MatrixView(dout + (size_t)r0 * n, r1 - r0, n, n, 1),
                       Bt,
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
    unsigned char *Ae = TensorEntries(op->A);
    size_t aSize = DataSize(TensorDataType(op->A));
    Scale aScale = ScaleGet(op->A);

    Matrix At  = MatrixView((Real *)Ae, k, end - start, aColStride, aRowStride);
    At.type    = TensorDataType(op->A);
    At.scale   = aScale;

    if (batchStrideB == 0 && batchStrideA == m * aRowStride && batchSize > 1 && (dimsA < 3 || !aScale.values)) {
        Real local[(size_t)k * n];
        memset(local, 0, (size_t)k * n * sizeof(Real));
        At.data = (Real *)(Ae + (size_t)start * aRowStride * aSize);
        At.scale.values = aScale.values ? aScale.values + start : NULL;
        At.scale.zeros  = aScale.zeros  ? aScale.zeros  + start : NULL;
        MatrixMultiply(At,
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
            Real *dout  = TensorGradients(op->out) + (size_t)batch * m * n;
            Real *dB    = TensorGradients(op->B)   + (size_t)batch * batchStrideB;
            int aGroup  = dimsA >= 3 ? batch : 0;

            At.data = (Real *)(Ae + ((size_t)batch * batchStrideA + (size_t)r0 * aColStride) * aSize);
            At.rows = r1 - r0;
            At.cols = m;
            At.scale.values = aScale.values ? aScale.values + aGroup : NULL;
            At.scale.zeros  = aScale.zeros  ? aScale.zeros  + aGroup : NULL;
            MatrixMultiply(At,
                           MatrixView(dout, m, n, n, 1),
                           MatrixView(dB + (size_t)r0 * bRowStride, r1 - r0, n, bRowStride, bColStride));
            i = chunkEnd;
        }
    } else if (batchStrideA == m * aRowStride && (dimsA < 3 || !aScale.values)) {
        // If A's batches are contiguous, fuse into a single GEMM
        At.data = (Real *)(Ae + (size_t)start * aColStride * aSize);
        At.rows = end - start;
        At.cols = batchSize * m;
        MatrixMultiply(At,
                       MatrixView(TensorGradients(op->out), batchSize * m, n, n, 1),
                       MatrixView(TensorGradients(op->B) + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride));
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Real *dout = TensorGradients(op->out) + (size_t)batch * m * n;
            int aGroup = dimsA >= 3 ? batch : 0;
            At.data = (Real *)(Ae + ((size_t)batch * batchStrideA + (size_t)start * aColStride) * aSize);
            At.rows = end - start;
            At.cols = m;
            At.scale.values = aScale.values ? aScale.values + aGroup : NULL;
            At.scale.zeros  = aScale.zeros  ? aScale.zeros  + aGroup : NULL;
            MatrixMultiply(At,
                           MatrixView(dout, m, n, n, 1),
                           MatrixView(TensorGradients(op->B) + (size_t)start * bRowStride, end - start, n, bRowStride, bColStride));
        }
    }
}
