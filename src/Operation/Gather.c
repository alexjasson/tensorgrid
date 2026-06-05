// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"

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
    Tensor out = TensorNew(outDims, outShape, A.gradients != NULL);
    return OperationExecute(OperationNew(out, A, B, OperationGatherForward, OperationGatherBackward, ctx));
}

static void OperationGatherForward(Operation *op)
{
    OperationDispatch(gatherForward, op, TensorSize(op->out), TensorSize(op->out));
}

static void OperationGatherBackward(Operation *op)
{
    if (!op->A.gradients) return;
    OperationDispatch(gatherBackward, op, TensorSize(op->A), (long)TensorSize(op->A) * TensorSize(op->B));
}


static void gatherForward(void *arg, int start, int end)
{
    Operation *op = arg;
    GatherContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
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
    GatherContext *ctx = op->context;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = ctx->axis;
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
