// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../math/Scalar.h"
#include "../utility.h"
#include "../math/Vector.h"

typedef struct {
    Binary b;
    Entry fill;
    int axis;
} ScatterContext;

static void OperationScatterForward(Operation *op);
static void OperationScatterBackward(Operation *op);
static void scatterForward(void *arg, int start, int end);
static void scatterBackward(void *arg, int start, int end);

Tensor OperationScatter(Binary fn, Tensor A, Tensor B, Entry fill, int axis, int n)
{
    int outDims = A.dimensions - B.dimensions + 1;
    int outShape[MAX_DIMENSIONS];
    int p = 0;
    for (int d = 0; d < axis; d++) outShape[p++] = A.shape[d];
    outShape[p++] = n;
    for (int d = axis + B.dimensions; d < A.dimensions; d++) outShape[p++] = A.shape[d];

    ScatterContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->b = fn;
    ctx->fill = fill;
    ctx->axis = axis;
    Tensor out = TensorNew(outDims, outShape, A.gradients != NULL);
    return OperationExecute(OperationNew(out, A, B, OperationScatterForward, OperationScatterBackward, ctx));
}

static void OperationScatterForward(Operation *op)
{
    OperationDispatch(scatterForward, op, TensorSize(op->out), (long)TensorSize(op->out) * TensorSize(op->B));
}

static void OperationScatterBackward(Operation *op)
{
    if (!op->A.gradients) return;
    OperationDispatch(scatterBackward, op, TensorSize(op->out), (long)TensorSize(op->out) * TensorSize(op->B));
}


static void scatterForward(void *arg, int start, int end)
{
    Operation *op = arg;
    ScatterContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = ctx->axis;
    int n         = op->out.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) innerSize *= op->out.shape[d];
    int aInnerDims = A.dimensions - axis - I.dimensions;
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    // Vectorized MAX/MIN/ADD reduction of each output's contiguous input segment
    f reducer = ctx->b.f;
    if (n == 1 && innerSize == 1 && aC &&
        (reducer == MAX.f || reducer == MIN.f || reducer == ADD.f)) {
        Entry *out = op->out.entries;
        Entry  fill = ctx->fill;
        for (int i = start; i < end; i++) {
            const Entry *row = A.entries + (size_t)i * idxSize;
            int j = 0;
            if (reducer == ADD.f) {
                Vector acc = VectorZero();
                for (; j + VECTOR_DIMS <= idxSize; j += VECTOR_DIMS)
                    acc = VectorAdd(acc, VectorLoad(row + j));
                Entry r = fill + VectorHsum(acc);
                for (; j < idxSize; j++) r += row[j];
                out[i] = r;
            } else {
                Vector acc = VectorBcast(fill);
                for (; j + VECTOR_DIMS <= idxSize; j += VECTOR_DIMS)
                    acc = (reducer == MAX.f) ? VectorMax(acc, VectorLoad(row + j))
                                             : VectorMin(acc, VectorLoad(row + j));
                Entry lanes[VECTOR_DIMS]; VectorStore(lanes, acc);
                Entry r = fill;
                for (int t = 0; t < VECTOR_DIMS; t++)
                    r = (reducer == MAX.f) ? (lanes[t] > r ? lanes[t] : r)
                                           : (lanes[t] < r ? lanes[t] : r);
                for (; j < idxSize; j++)
                    r = (reducer == MAX.f) ? (row[j] > r ? row[j] : r)
                                           : (row[j] < r ? row[j] : r);
                out[i] = r;
            }
        }
        return;
    }

    if (aC && iC) {
        for (int i = start; i < end; i++) {
            int inner   = i % innerSize;
            int rem     = i / innerSize;
            int kTarget = rem % n;
            int outer   = rem / n;

            Entry acc = ctx->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)I.entries[ix] == kTarget)
                    acc = ((ScalarBinary)ctx->b.f)(acc, A.entries[(outer * idxSize + ix) * innerSize + inner]);
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

            Entry acc = ctx->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
                if ((int)I.entries[idxOff] == kTarget) {
                    int aIxOff = aC ? ix * innerSize
                                    : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                    acc = ((ScalarBinary)ctx->b.f)(acc, A.entries[outerOff + aIxOff + innerOff]);
                }
            }
            op->out.entries[i] = acc;
        }
    }
}

static void scatterBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    ScatterContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = ctx->axis;
    int n         = op->out.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) innerSize *= op->out.shape[d];
    int aInnerDims = A.dimensions - axis - I.dimensions;
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    // ADD's gradient is 1, so each output gradient adds into its input segment
    if (n == 1 && innerSize == 1 && aC && ctx->b.f == ADD.f) {
        Entry *dout = op->out.gradients;
        for (int i = start; i < end; i++) {
            Vector gv = VectorBcast(dout[i]);
            Entry *row = A.gradients + (size_t)i * idxSize;
            int j = 0;
            for (; j + VECTOR_DIMS <= idxSize; j += VECTOR_DIMS)
                VectorStore(row + j, VectorAdd(VectorLoad(row + j), gv));
            for (; j < idxSize; j++) row[j] += dout[i];
        }
        return;
    }

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
        chain[0] = ctx->fill;
        int count = 0;
        for (int ix = 0; ix < idxSize; ix++) {
            int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
            if ((int)I.entries[idxOff] == kTarget) {
                int aIxOff = aC ? ix * innerSize
                                : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                Entry Av = A.entries[outerOff + aIxOff + innerOff];
                chain[count + 1] = ((ScalarBinary)ctx->b.f)(chain[count], Av);
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
                A.gradients[aOffset] += dresult * ((ScalarBinary)ctx->b.db)(chain[c], Av);
                dresult *= ((ScalarBinary)ctx->b.da)(chain[c], Av);
                c--;
            }
        }
    }
}
