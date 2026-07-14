// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Vector.h"
#include "../math/Scalar.h"

typedef struct { int axis; } GatherContext;

static void OperationGatherForward(Operation *op);
static void OperationGatherBackward(Operation *op);
static void gatherForward(void *arg, int start, int end);
static void gatherBackward(void *arg, int start, int end);

Tensor OperationGather(Tensor A, Tensor B, int axis)
{
    int outDims = A.dimensions - 1 + B.dimensions;
    int outShape[MAX_DIMENSIONS];
    int p = 0;
    for (int d = 0; d < axis; d++) outShape[p++] = A.shape[d];
    for (int d = 0; d < B.dimensions; d++) outShape[p++] = B.shape[d];
    for (int d = axis + 1; d < A.dimensions; d++) outShape[p++] = A.shape[d];

    GatherContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->axis = axis;
    Tensor out = TensorNew(outDims, outShape, REAL, A.gradients != NULL);
    return OperationExecute(OperationNew(out, A, B, OperationGatherForward, OperationGatherBackward, ctx));
}

static void OperationGatherForward(Operation *op)
{
    OperationDispatch(gatherForward, op, TensorSize(op->out), TensorSize(op->out));
}

static void OperationGatherBackward(Operation *op)
{
    if (!op->A.gradients) return;
    GatherContext *ctx = op->context;
    Tensor A = op->A, I = op->B;
    int innerSize = 1;
    for (int d = ctx->axis + 1; d < A.dimensions; d++) innerSize *= A.shape[d];
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    if (aC && iC && innerSize >= VECTOR_DIMS)
        OperationDispatch(gatherBackward, op, innerSize, TensorSize(op->out));
    else
        OperationDispatch(gatherBackward, op, TensorSize(op->A), (long)TensorSize(op->A) * TensorSize(op->B));
}


static void gatherForward(void *arg, int start, int end)
{
    Operation *op = arg;
    GatherContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
    Real *oute    = op->out.entries;
    int axis      = ctx->axis;
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
            int k     = (int)(TensorGet(I, ix) + (Real)0.5);
            oute[i] = TensorGet(A, (outer * axisSize + k) * innerSize + inner);
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
            int k        = (int)(TensorGet(I, idxOff) + (Real)0.5);
            oute[i] = TensorGet(A, outerOff + k * aStride + innerOff);
        }
    }
}

static void gatherBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    GatherContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
    Real *Ie      = I.entries;
    int axis      = ctx->axis;
    int axisSize  = A.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < A.dimensions; d++) innerSize *= A.shape[d];
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    // Fast path to parallelize over the inner columns
    if (aC && iC && innerSize >= VECTOR_DIMS) {
        int outer = 1;
        for (int d = 0; d < axis; d++) outer *= A.shape[d];
        Real *dA   = A.gradients;
        Real *dout = op->out.gradients;
        for (int o = 0; o < outer; o++) {
            for (int ix = 0; ix < idxSize; ix++) {
                int k = (int)(Ie[ix] + (Real)0.5);
                const Real *src = dout + ((size_t)o * idxSize  + ix) * innerSize;
                Real       *dst = dA   + ((size_t)o * axisSize + k)  * innerSize;
                int c = start;
                for (; c + VECTOR_DIMS <= end; c += VECTOR_DIMS)
                    VectorStore(dst + c, VectorAdd(VectorLoad(dst + c, REAL), VectorLoad(src + c, REAL)));
                for (; c < end; c++)
                    dst[c] += src[c];
            }
        }
        return;
    }

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

        Real sum = 0;
        if (iC) {
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)(Ie[ix] + (Real)0.5) == kTarget)
                    sum += op->out.gradients[(outer * idxSize + ix) * innerSize + inner];
            }
        } else {
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = unravel(ix, I.shape, I.strides, I.dimensions);
                if ((int)(Ie[idxOff] + (Real)0.5) == kTarget)
                    sum += op->out.gradients[(outer * idxSize + ix) * innerSize + inner];
            }
        }
        A.gradients[outerOff + kTarget * aStride + innerOff] += sum;
    }
}
