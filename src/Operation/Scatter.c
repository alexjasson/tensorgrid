// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Vector.h"

typedef struct {
    Binary b;
    Real fill;
    int axis;
} ScatterContext;

static void OperationScatterForward(Operation *op);
static void OperationScatterBackward(Operation *op);
static void scatterForward(void *arg, int start, int end);
static void scatterBackward(void *arg, int start, int end);

Tensor OperationScatter(Binary fn, Tensor A, Tensor B, Real fill, int axis, int n)
{
    int outDims = TensorDimensions(A) - TensorDimensions(B) + 1;
    int outShape[MAX_DIMENSIONS];
    int p = 0;
    for (int d = 0; d < axis; d++) outShape[p++] = TensorShape(A)[d];
    outShape[p++] = n;
    for (int d = axis + TensorDimensions(B); d < TensorDimensions(A); d++) outShape[p++] = TensorShape(A)[d];

    ScatterContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->b = fn;
    ctx->fill = fill;
    ctx->axis = axis;
    Tensor out = TensorNew(outDims, outShape, FLOAT32, TensorGradients(A) != NULL);
    return OperationExecute(OperationNew(out, A, B, OperationScatterForward, OperationScatterBackward, ctx));
}

static void OperationScatterForward(Operation *op)
{
    OperationDispatch(scatterForward, op, TensorSize(op->out), (long)TensorSize(op->out) * TensorSize(op->B));
}

static void OperationScatterBackward(Operation *op)
{
    if (!TensorGradients(op->A)) return;
    OperationDispatch(scatterBackward, op, TensorSize(op->out), (long)TensorSize(op->out) * TensorSize(op->B));
}


static void scatterForward(void *arg, int start, int end)
{
    Operation *op = arg;
    ScatterContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
    Real *oute    = TensorEntries(op->out);
    int axis      = ctx->axis;
    int n         = TensorShape(op->out)[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < TensorDimensions(op->out); d++) innerSize *= TensorShape(op->out)[d];
    int aInnerDims = TensorDimensions(A) - axis - TensorDimensions(I);
    int aC = isContiguous(TensorShape(A), TensorStride(A), TensorDimensions(A));
    int iC = isContiguous(TensorShape(I), TensorStride(I), TensorDimensions(I));

    // Vectorized MAX/MIN/ADD reduction of each output's contiguous input segment
    g reducer = ctx->b.g;
    if (n == 1 && innerSize == 1 && aC && TensorDataType(A) == FLOAT32 && !TensorScale(A) &&
        (reducer == MAX.g || reducer == MIN.g || reducer == ADD.g)) {
        Real *out = TensorEntries(op->out);
        Real  fill = ctx->fill;
        for (int i = start; i < end; i++) {
            const Real *row = (const Real *)TensorEntries(A) + (size_t)i * idxSize;
            int j = 0;
            if (reducer == ADD.g) {
                Vector acc = VectorZero();
                for (; j + VECTOR_DIMS <= idxSize; j += VECTOR_DIMS)
                    acc = VectorAdd(acc, VectorLoad(row + j, FLOAT32));
                Real r = fill + VectorHsum(acc);
                for (; j < idxSize; j++) r += row[j];
                out[i] = r;
            } else {
                Vector acc = VectorBcast(fill);
                for (; j + VECTOR_DIMS <= idxSize; j += VECTOR_DIMS)
                    acc = (reducer == MAX.g) ? VectorMax(acc, VectorLoad(row + j, FLOAT32))
                                             : VectorMin(acc, VectorLoad(row + j, FLOAT32));
                Real lanes[VECTOR_DIMS]; VectorStore(lanes, acc);
                Real r = fill;
                for (int t = 0; t < VECTOR_DIMS; t++)
                    r = (reducer == MAX.g) ? (lanes[t] > r ? lanes[t] : r)
                                           : (lanes[t] < r ? lanes[t] : r);
                for (; j < idxSize; j++)
                    r = (reducer == MAX.g) ? (row[j] > r ? row[j] : r)
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

            Real acc = ctx->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)(TensorGet(I, ix) + (Real)0.5) == kTarget)
                    acc = VectorApply((VectorBinary)ctx->b.g, acc, TensorGet(A, (outer * idxSize + ix) * innerSize + inner));
            }
            oute[i] = acc;
        }
    } else {
        for (int i = start; i < end; i++) {
            int inner   = i % innerSize;
            int rem     = i / innerSize;
            int kTarget = rem % n;
            int outer   = rem / n;

            int outerOff = aC ? outer * idxSize * innerSize
                              : unravel(outer, TensorShape(A), TensorStride(A), axis);
            int innerOff = aC ? inner
                              : unravel(inner, TensorShape(A) + axis + TensorDimensions(I),
                                        TensorStride(A) + axis + TensorDimensions(I), aInnerDims);

            Real acc = ctx->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = iC ? ix : unravel(ix, TensorShape(I), TensorStride(I), TensorDimensions(I));
                if ((int)(TensorGet(I, idxOff) + (Real)0.5) == kTarget) {
                    int aIxOff = aC ? ix * innerSize
                                    : unravel(ix, TensorShape(A) + axis, TensorStride(A) + axis, TensorDimensions(I));
                    acc = VectorApply((VectorBinary)ctx->b.g, acc, TensorGet(A, outerOff + aIxOff + innerOff));
                }
            }
            oute[i] = acc;
        }
    }
}

static void scatterBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    ScatterContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
    Real *Ae      = TensorEntries(A);
    Real *Ie      = TensorEntries(I);
    int axis      = ctx->axis;
    int n         = TensorShape(op->out)[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < TensorDimensions(op->out); d++) innerSize *= TensorShape(op->out)[d];
    int aInnerDims = TensorDimensions(A) - axis - TensorDimensions(I);
    int aC = isContiguous(TensorShape(A), TensorStride(A), TensorDimensions(A));
    int iC = isContiguous(TensorShape(I), TensorStride(I), TensorDimensions(I));

    // ADD's gradient is 1, so each output gradient adds into its input segment
    if (n == 1 && innerSize == 1 && aC && ctx->b.g == ADD.g) {
        Real *dout = TensorGradients(op->out);
        for (int i = start; i < end; i++) {
            Vector gv = VectorBcast(dout[i]);
            Real *row = TensorGradients(A) + (size_t)i * idxSize;
            int j = 0;
            for (; j + VECTOR_DIMS <= idxSize; j += VECTOR_DIMS)
                VectorStore(row + j, VectorAdd(VectorLoad(row + j, FLOAT32), gv));
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
            outerOff = unravel(outer, TensorShape(A), TensorStride(A), axis);
            innerOff = unravel(inner, TensorShape(A) + axis + TensorDimensions(I),
                               TensorStride(A) + axis + TensorDimensions(I), aInnerDims);
        }

        Real chain[idxSize + 1];
        chain[0] = ctx->fill;
        int count = 0;
        for (int ix = 0; ix < idxSize; ix++) {
            int idxOff = iC ? ix : unravel(ix, TensorShape(I), TensorStride(I), TensorDimensions(I));
            if ((int)(Ie[idxOff] + (Real)0.5) == kTarget) {
                int aIxOff = aC ? ix * innerSize
                                : unravel(ix, TensorShape(A) + axis, TensorStride(A) + axis, TensorDimensions(I));
                Real Av = Ae[outerOff + aIxOff + innerOff];
                chain[count + 1] = VectorApply((VectorBinary)ctx->b.g, chain[count], Av);
                count++;
            }
        }

        Real dresult = TensorGradients(op->out)[i];
        int c = count - 1;
        for (int ix = idxSize - 1; ix >= 0 && c >= 0; ix--) {
            int idxOff = iC ? ix : unravel(ix, TensorShape(I), TensorStride(I), TensorDimensions(I));
            if ((int)(Ie[idxOff] + (Real)0.5) == kTarget) {
                int aIxOff  = aC ? ix * innerSize
                                 : unravel(ix, TensorShape(A) + axis, TensorStride(A) + axis, TensorDimensions(I));
                int aOffset = outerOff + aIxOff + innerOff;
                Real Av     = Ae[aOffset];
                TensorGradients(A)[aOffset] += dresult * VectorApply((VectorBinary)ctx->b.db, chain[c], Av);
                dresult *= VectorApply((VectorBinary)ctx->b.da, chain[c], Av);
                c--;
            }
        }
    }
}
