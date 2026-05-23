// https://github.com/alexjasson/tensorgrid

#include "Operation.h"
#include "utility.h"
#include "ThreadPool.h"

// Register tile sizes for the multiply kernel
#if defined(__AVX512F__)
#define MR 6
#define NR 16
#elif defined(__AVX2__)
#define MR 4
#define NR 8
#else
#define MR 2
#define NR 4
#endif

// Forward kernels
static void mapForward(void *arg, int start, int end);
static void zipForward(void *arg, int start, int end);
static void multiplyForward(void *arg, int start, int end);
static void unfoldForward(void *arg, int start, int end);
static void gatherForward(void *arg, int start, int end);
static void scatterForward(void *arg, int start, int end);

// Backward kernels
static void mapBackward(void *arg, int start, int end);
static void zipBackward(void *arg, int start, int end);
static void multiplyBackwardDa(void *arg, int start, int end);
static void multiplyBackwardDb(void *arg, int start, int end);
static void unfoldBackward(void *arg, int start, int end);
static void gatherBackward(void *arg, int start, int end);
static void scatterBackward(void *arg, int start, int end);

static void mapForward(void *arg, int start, int end)
{
    Operation *op = arg;
    if (isContiguous(op->A.shape, op->A.strides, op->A.dimensions)) {
        for (int i = start; i < end; i++)
            op->out.entries[i] = op->u.f(op->A.entries[i]);
    } else {
        for (int i = start; i < end; i++) {
            int aOff = unravel(i, op->A.shape, op->A.strides, op->A.dimensions);
            op->out.entries[i] = op->u.f(op->A.entries[aOff]);
        }
    }
}

static void zipForward(void *arg, int start, int end)
{
    Operation *op = arg;
    int aC = isContiguous(op->A.shape, op->A.strides, op->A.dimensions);
    int bC = isContiguous(op->B.shape, op->B.strides, op->B.dimensions);
    for (int i = start; i < end; i++) {
        int aOff = aC ? i : unravel(i, op->A.shape, op->A.strides, op->A.dimensions);
        int bOff = bC ? i : unravel(i, op->B.shape, op->B.strides, op->B.dimensions);
        op->out.entries[i] = op->b.f(op->A.entries[aOff], op->B.entries[bOff]);
    }
}

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
        int row        = i % m;
        int batchEnd   = (batch + 1) * m;
        int chunkEnd   = (end < batchEnd) ? end : batchEnd;
        int chunkLen   = chunkEnd - i;
        Entry *A_batch = op->A.entries   + batch * batchStrideA;
        Entry *B_batch = op->B.entries   + batch * batchStrideB;
        Entry *C_batch = op->out.entries + batch * m * n;

        int blocked = chunkLen - (chunkLen % MR);
        int r = 0;
        for (; r < blocked; r += MR) {
            Entry *C[MR];
            for (int mr = 0; mr < MR; mr++)
                C[mr] = C_batch + (row + r + mr) * n;
            for (int j = 0; j < n; j++)
                for (int mr = 0; mr < MR; mr++)
                    C[mr][j] = 0.0f;
            for (int l = 0; l < k; l++) {
                Entry a[MR];
                for (int mr = 0; mr < MR; mr++)
                    a[mr] = A_batch[(row + r + mr) * strideA_row + l * strideA_col];
                int bBase = l * strideB_row;
                for (int j = 0; j < n; j++) {
                    Entry b = B_batch[bBase + j * strideB_col];
                    for (int mr = 0; mr < MR; mr++)
                        C[mr][j] += a[mr] * b;
                }
            }
        }
        for (; r < chunkLen; r++) {
            Entry *C_row = C_batch + (row + r) * n;
            for (int j = 0; j < n; j++) C_row[j] = 0.0f;
            for (int l = 0; l < k; l++) {
                Entry a = A_batch[(row + r) * strideA_row + l * strideA_col];
                int bBase = l * strideB_row;
                for (int j = 0; j < n; j++)
                    C_row[j] += a * B_batch[bBase + j * strideB_col];
            }
        }
        i = chunkEnd;
    }
}

static void mapBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *A    = op->A.entries;
    Entry *dA   = op->A.gradients;
    Entry *dout = op->out.gradients;
    if (isContiguous(op->A.shape, op->A.strides, op->A.dimensions)) {
        for (int j = start; j < end; j++)
            dA[j] += dout[j] * op->u.da(A[j]);
    } else {
        for (int j = start; j < end; j++) {
            int aOff = unravel(j, op->A.shape, op->A.strides, op->A.dimensions);
            dA[aOff] += dout[j] * op->u.da(A[aOff]);
        }
    }
}

static void zipBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *A    = op->A.entries;
    Entry *B    = op->B.entries;
    Entry *dA   = op->A.gradients;
    Entry *dB   = op->B.gradients;
    Entry *dout = op->out.gradients;
    int aC = isContiguous(op->A.shape, op->A.strides, op->A.dimensions);
    int bC = isContiguous(op->B.shape, op->B.strides, op->B.dimensions);
    for (int j = start; j < end; j++) {
        int aOff = aC ? j : unravel(j, op->A.shape, op->A.strides, op->A.dimensions);
        int bOff = bC ? j : unravel(j, op->B.shape, op->B.strides, op->B.dimensions);
        Entry g = dout[j];
        if (dA) dA[aOff] += g * op->b.da(A[aOff], B[bOff]);
        if (dB) dB[bOff] += g * op->b.db(A[aOff], B[bOff]);
    }
}

// dA = dout @ B^T
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

    int i = start;
    while (i < end) {
        int batchFrom, batchTo, row, chunkEnd;
        if (batchStrideA) {
            int batch = i / m;
            batchFrom = batch;
            batchTo   = batch + 1;
            row       = i % m;
            int batchEndGlobal = (batch + 1) * m;
            chunkEnd  = (end < batchEndGlobal) ? end : batchEndGlobal;
        } else {
            batchFrom = 0;
            batchTo   = batchSize;
            row       = i;
            chunkEnd  = end;
        }
        int chunkLen = chunkEnd - i;
        Entry *dA = op->A.gradients + batchFrom * batchStrideA;

        int blocked = chunkLen - (chunkLen % MR);
        int r = 0;
        for (; r < blocked; r += MR) {
            for (int col = 0; col < k; col++) {
                Entry s[MR];
                for (int mr = 0; mr < MR; mr++) s[mr] = 0;
                for (int batch = batchFrom; batch < batchTo; batch++) {
                    Entry *B    = op->B.entries     + batch * batchStrideB;
                    Entry *dout = op->out.gradients + batch * m * n;
                    Entry *dout_rows[MR];
                    for (int mr = 0; mr < MR; mr++)
                        dout_rows[mr] = dout + (row + r + mr) * n;
                    int bBase = col * bRowStride;
                    for (int j = 0; j < n; j++) {
                        Entry b = B[bBase + j * bColStride];
                        for (int mr = 0; mr < MR; mr++)
                            s[mr] += dout_rows[mr][j] * b;
                    }
                }
                for (int mr = 0; mr < MR; mr++)
                    dA[(row + r + mr) * aRowStride + col * aColStride] += s[mr];
            }
        }
        for (; r < chunkLen; r++) {
            for (int col = 0; col < k; col++) {
                Entry sum = 0;
                for (int batch = batchFrom; batch < batchTo; batch++) {
                    Entry *B    = op->B.entries     + batch * batchStrideB;
                    Entry *dout = op->out.gradients + batch * m * n;
                    int bBase = col * bRowStride;
                    for (int j = 0; j < n; j++)
                        sum += dout[(row + r) * n + j] * B[bBase + j * bColStride];
                }
                dA[(row + r) * aRowStride + col * aColStride] += sum;
            }
        }

        i = chunkEnd;
    }
}

// dB = A^T @ dout
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

    for (int i = start; i < end; i++) {
        int batchFrom, batchTo, row;
        if (batchStrideB) { batchFrom = i / k; batchTo = batchFrom + 1; row = i % k; }
        else              { batchFrom = 0;     batchTo = batchSize;     row = i;     }

        Entry *dB = op->B.gradients + batchFrom * batchStrideB;

        int blocked = n - (n % NR);
        int c = 0;
        for (; c < blocked; c += NR) {
            Entry sums[NR] = {0};
            for (int batch = batchFrom; batch < batchTo; batch++) {
                Entry *A    = op->A.entries     + batch * batchStrideA;
                Entry *dout = op->out.gradients + batch * m * n;
                for (int j = 0; j < m; j++) {
                    Entry a = A[j * aRowStride + row * aColStride];
                    Entry *dout_row = dout + j * n + c;
                    for (int nr = 0; nr < NR; nr++)
                        sums[nr] += a * dout_row[nr];
                }
            }
            for (int nr = 0; nr < NR; nr++)
                dB[row * bRowStride + (c + nr) * bColStride] += sums[nr];
        }
        for (; c < n; c++) {
            Entry sum = 0;
            for (int batch = batchFrom; batch < batchTo; batch++) {
                Entry *A    = op->A.entries     + batch * batchStrideA;
                Entry *dout = op->out.gradients + batch * m * n;
                for (int j = 0; j < m; j++)
                    sum += A[j * aRowStride + row * aColStride] * dout[j * n + c];
            }
            dB[row * bRowStride + c * bColStride] += sum;
        }
    }
}

static void unfoldForward(void *arg, int start, int end)
{
    Operation *op = arg;
    int Cin    = op->A.shape[1];
    int H      = op->A.shape[2];
    int W      = op->A.shape[3];
    int fH     = op->fH,     fW      = op->fW;
    int stride = op->stride, padding = op->padding;
    int Hout   = (H + 2 * padding - fH) / stride + 1;
    int Wout   = (W + 2 * padding - fW) / stride + 1;
    int S      = Hout * Wout;
    int K      = Cin * fH * fW;
    int sN     = op->A.strides[0];
    int sC     = op->A.strides[1];
    int sH     = op->A.strides[2];
    int sW     = op->A.strides[3];
    Entry        fill    = op->fill;
    const Entry *input   = op->A.entries;
    Entry       *windows = op->out.entries;

    for (int i = start; i < end; i++) {
        int n  = i / K;
        int k  = i % K;
        int ci = k / (fH * fW);
        int fh = (k / fW) % fH;
        int fw = k % fW;
        Entry       *windowRow = windows + (n * K + k) * S;
        const Entry *inPlane   = input + n * sN + ci * sC;

        for (int oh = 0; oh < Hout; oh++) {
            int ih = oh * stride - padding + fh;
            if (ih < 0 || ih >= H) {
                for (int ow = 0; ow < Wout; ow++)
                    windowRow[oh * Wout + ow] = fill;
            } else {
                const Entry *row = inPlane + ih * sH;
                for (int ow = 0; ow < Wout; ow++) {
                    int iw = ow * stride - padding + fw;
                    windowRow[oh * Wout + ow] = (iw >= 0 && iw < W) ? row[iw * sW] : fill;
                }
            }
        }
    }
}

static void unfoldBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    int Cin    = op->A.shape[1];
    int H      = op->A.shape[2];
    int W      = op->A.shape[3];
    int fH     = op->fH,     fW      = op->fW;
    int stride = op->stride, padding = op->padding;
    int Hout   = (H + 2 * padding - fH) / stride + 1;
    int Wout   = (W + 2 * padding - fW) / stride + 1;
    int S      = Hout * Wout;
    int K      = Cin * fH * fW;
    int sN     = op->A.strides[0];
    int sC     = op->A.strides[1];
    int sH     = op->A.strides[2];
    int sW     = op->A.strides[3];
    Entry       *dInput  = op->A.gradients;
    const Entry *dWindow = op->out.gradients;

    for (int i = start; i < end; i++) {
        int n  = i / (Cin * H * W);
        int ci = (i / (H * W)) % Cin;
        int ih = (i / W) % H;
        int iw = i % W;
        const Entry *dP_n = dWindow + n * K * S;

        Entry sum = 0;
        for (int fh = 0; fh < fH; fh++) {
            int numH = ih + padding - fh;
            if (numH < 0 || numH % stride) continue;
            int oh = numH / stride;
            if (oh >= Hout) continue;
            for (int fw = 0; fw < fW; fw++) {
                int numW = iw + padding - fw;
                if (numW < 0 || numW % stride) continue;
                int ow = numW / stride;
                if (ow >= Wout) continue;
                int k = (ci * fH + fh) * fW + fw;
                sum += dP_n[k * S + oh * Wout + ow];
            }
        }
        dInput[n * sN + ci * sC + ih * sH + iw * sW] += sum;
    }
}

static void gatherForward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int axisSize  = A.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < A.dimensions; d++) innerSize *= A.shape[d];
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    if (aC && iC) {
        for (int i = start; i < end; i++) {
            int inner = i % innerSize;
            int rem   = i / innerSize;
            int ix    = rem % idxSize;
            int outer = rem / idxSize;
            int k     = (int)I.entries[ix];
            op->out.entries[i] = A.entries[(outer * axisSize + k) * innerSize + inner];
        }
    } else {
        for (int i = start; i < end; i++) {
            int inner = i % innerSize;
            int rem   = i / innerSize;
            int ix    = rem % idxSize;
            int outer = rem / idxSize;

            int outerOff = aC ? outer * axisSize * innerSize
                              : unravel(outer, A.shape, A.strides, axis);
            int innerOff = aC ? inner
                              : unravel(inner, A.shape + axis + 1, A.strides + axis + 1,
                                        A.dimensions - axis - 1);
            int aStride  = aC ? innerSize : A.strides[axis];
            int idxOff   = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
            int k        = (int)I.entries[idxOff];
            op->out.entries[i] = A.entries[outerOff + k * aStride + innerOff];
        }
    }
}

static void gatherBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int axisSize  = A.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < A.dimensions; d++) innerSize *= A.shape[d];
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    for (int i = start; i < end; i++) {
        int inner   = i % innerSize;
        int rem     = i / innerSize;
        int kTarget = rem % axisSize;
        int outer   = rem / axisSize;

        int outerOff, innerOff, aStride;
        if (aC) {
            outerOff = outer * axisSize * innerSize;
            innerOff = inner;
            aStride  = innerSize;
        } else {
            outerOff = unravel(outer, A.shape, A.strides, axis);
            innerOff = unravel(inner, A.shape + axis + 1, A.strides + axis + 1,
                               A.dimensions - axis - 1);
            aStride  = A.strides[axis];
        }

        Entry sum = 0;
        if (iC) {
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)I.entries[ix] == kTarget)
                    sum += op->out.gradients[(outer * idxSize + ix) * innerSize + inner];
            }
        } else {
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = unravel(ix, I.shape, I.strides, I.dimensions);
                if ((int)I.entries[idxOff] == kTarget)
                    sum += op->out.gradients[(outer * idxSize + ix) * innerSize + inner];
            }
        }
        A.gradients[outerOff + kTarget * aStride + innerOff] += sum;
    }
}

static void scatterForward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int n         = op->out.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) innerSize *= op->out.shape[d];
    int aInnerDims = A.dimensions - axis - I.dimensions;
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    if (aC && iC) {
        for (int i = start; i < end; i++) {
            int inner   = i % innerSize;
            int rem     = i / innerSize;
            int kTarget = rem % n;
            int outer   = rem / n;

            Entry acc = op->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)I.entries[ix] == kTarget)
                    acc = op->b.f(acc, A.entries[(outer * idxSize + ix) * innerSize + inner]);
            }
            op->out.entries[i] = acc;
        }
    } else {
        for (int i = start; i < end; i++) {
            int inner   = i % innerSize;
            int rem     = i / innerSize;
            int kTarget = rem % n;
            int outer   = rem / n;

            int outerOff = aC ? outer * idxSize * innerSize
                              : unravel(outer, A.shape, A.strides, axis);
            int innerOff = aC ? inner
                              : unravel(inner, A.shape + axis + I.dimensions,
                                        A.strides + axis + I.dimensions, aInnerDims);

            Entry acc = op->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
                if ((int)I.entries[idxOff] == kTarget) {
                    int aIxOff = aC ? ix * innerSize
                                    : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                    acc = op->b.f(acc, A.entries[outerOff + aIxOff + innerOff]);
                }
            }
            op->out.entries[i] = acc;
        }
    }
}

static void scatterBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int n         = op->out.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) innerSize *= op->out.shape[d];
    int aInnerDims = A.dimensions - axis - I.dimensions;
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    for (int i = start; i < end; i++) {
        int inner   = i % innerSize;
        int rem     = i / innerSize;
        int kTarget = rem % n;
        int outer   = rem / n;

        int outerOff, innerOff;
        if (aC) {
            outerOff = outer * idxSize * innerSize;
            innerOff = inner;
        } else {
            outerOff = unravel(outer, A.shape, A.strides, axis);
            innerOff = unravel(inner, A.shape + axis + I.dimensions,
                               A.strides + axis + I.dimensions, aInnerDims);
        }

        Entry chain[idxSize + 1];
        chain[0] = op->fill;
        int count = 0;
        for (int ix = 0; ix < idxSize; ix++) {
            int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
            if ((int)I.entries[idxOff] == kTarget) {
                int aIxOff = aC ? ix * innerSize
                                : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                Entry Av = A.entries[outerOff + aIxOff + innerOff];
                chain[count + 1] = op->b.f(chain[count], Av);
                count++;
            }
        }

        Entry dresult = op->out.gradients[i];
        int c = count - 1;
        for (int ix = idxSize - 1; ix >= 0 && c >= 0; ix--) {
            int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
            if ((int)I.entries[idxOff] == kTarget) {
                int aIxOff  = aC ? ix * innerSize
                                 : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                int aOffset = outerOff + aIxOff + innerOff;
                Entry Av    = A.entries[aOffset];
                A.gradients[aOffset] += dresult * op->b.db(chain[c], Av);
                dresult *= op->b.da(chain[c], Av);
                c--;
            }
        }
    }
}

void OperationForward(Operation op)
{
    ThreadPool pool = ThreadPoolGet();
    switch (op.type) {
        case Map:
            ThreadPoolMap(pool, mapForward, &op, TensorSize(op.out));
            break;
        case Zip:
            ThreadPoolMap(pool, zipForward, &op, TensorSize(op.out));
            break;
        case Multiply: {
            int m         = op.A.shape[op.A.dimensions - 2];
            int n         = op.B.shape[op.B.dimensions - 1];
            int batchSize = TensorSize(op.out) / (m * n);
            ThreadPoolMap(pool, multiplyForward, &op, batchSize * m);
            break;
        }
        case Unfold:
            ThreadPoolMap(pool, unfoldForward, &op, op.out.shape[0] * op.out.shape[1]);
            break;
        case Gather:
            ThreadPoolMap(pool, gatherForward, &op, TensorSize(op.out));
            break;
        case Scatter:
            ThreadPoolMap(pool, scatterForward, &op, TensorSize(op.out));
            break;
    }
}

void OperationBackward(Operation op)
{
    ThreadPool pool = ThreadPoolGet();
    switch (op.type) {
        case Map:
            if (op.A.gradients)
                ThreadPoolMap(pool, mapBackward, &op, TensorSize(op.out));
            break;
        case Zip:
            if (op.A.gradients || op.B.gradients)
                ThreadPoolMap(pool, zipBackward, &op, TensorSize(op.out));
            break;
        case Multiply: {
            int m            = op.A.shape[op.A.dimensions - 2];
            int k            = op.A.shape[op.A.dimensions - 1];
            int n            = op.B.shape[op.B.dimensions - 1];
            int batchSize    = TensorSize(op.out) / (m * n);
            int batchStrideA = (op.A.dimensions >= 3) ? op.A.strides[op.A.dimensions - 3] : 0;
            int batchStrideB = (op.B.dimensions >= 3) ? op.B.strides[op.B.dimensions - 3] : 0;
            if (op.A.gradients) {
                int total = batchStrideA ? batchSize * m : m;
                ThreadPoolMap(pool, multiplyBackwardDa, &op, total);
            }
            if (op.B.gradients) {
                int total = batchStrideB ? batchSize * k : k;
                ThreadPoolMap(pool, multiplyBackwardDb, &op, total);
            }
            break;
        }
        case Unfold:
            if (op.A.gradients)
                ThreadPoolMap(pool, unfoldBackward, &op, TensorSize(op.A));
            break;
        case Gather:
            if (op.A.gradients)
                ThreadPoolMap(pool, gatherBackward, &op, TensorSize(op.A));
            break;
        case Scatter:
            if (op.A.gradients)
                ThreadPoolMap(pool, scatterBackward, &op, TensorSize(op.out));
            break;
    }
}
