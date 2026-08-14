// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Vector.h"

typedef struct { int axis; } GatherContext;

static void OperationGatherForward(Operation *op);
static void OperationGatherBackward(Operation *op);
static void gatherForward(void *arg, int start, int end);
static void gatherBackward(void *arg, int start, int end);

Tensor OperationGather(Tensor A, Tensor B, int axis)
{
    int outDims = TensorDimensions(A) - 1 + TensorDimensions(B);
    int outShape[outDims > 0 ? outDims : 1];
    int p = 0;
    for (int d = 0; d < axis; d++) outShape[p++] = TensorShape(A)[d];
    for (int d = 0; d < TensorDimensions(B); d++) outShape[p++] = TensorShape(B)[d];
    for (int d = axis + 1; d < TensorDimensions(A); d++) outShape[p++] = TensorShape(A)[d];

    GatherContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->axis = axis;
    Tensor out = TensorNew(outDims, outShape, FLOAT32, TensorGradients(A) != NULL);
    return OperationExecute(OperationNew(out, A, B, OperationGatherForward, OperationGatherBackward, ctx));
}

static void OperationGatherForward(Operation *op)
{
    OperationDispatch(gatherForward, op, TensorSize(op->out), TensorSize(op->out));
}

static void OperationGatherBackward(Operation *op)
{
    if (!TensorGradients(op->A)) return;
    GatherContext *ctx = op->context;
    Tensor A = op->A, I = op->B;
    int innerSize = 1;
    for (int d = ctx->axis + 1; d < TensorDimensions(A); d++) innerSize *= TensorShape(A)[d];
    int aC = isContiguous(TensorShape(A), TensorStride(A), TensorDimensions(A));
    int iC = isContiguous(TensorShape(I), TensorStride(I), TensorDimensions(I));

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
    Real *oute    = TensorEntries(op->out);
    int axis      = ctx->axis;
    int axisSize  = TensorShape(A)[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < TensorDimensions(A); d++) innerSize *= TensorShape(A)[d];
    int aC = isContiguous(TensorShape(A), TensorStride(A), TensorDimensions(A));
    int iC = isContiguous(TensorShape(I), TensorStride(I), TensorDimensions(I));

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
                              : unravel(outer, TensorShape(A), TensorStride(A), axis);
            int innerOff = aC ? inner
                              : unravel(inner, TensorShape(A) + axis + 1, TensorStride(A) + axis + 1,
                                        TensorDimensions(A) - axis - 1);
            int aStride  = aC ? innerSize : TensorStride(A)[axis];
            int idxOff   = iC ? ix : unravel(ix, TensorShape(I), TensorStride(I), TensorDimensions(I));
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
    Real *Ie      = TensorEntries(I);
    int axis      = ctx->axis;
    int axisSize  = TensorShape(A)[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < TensorDimensions(A); d++) innerSize *= TensorShape(A)[d];
    int aC = isContiguous(TensorShape(A), TensorStride(A), TensorDimensions(A));
    int iC = isContiguous(TensorShape(I), TensorStride(I), TensorDimensions(I));

    // Fast path to parallelize over the inner columns
    if (aC && iC && innerSize >= VECTOR_DIMS) {
        int outer = 1;
        for (int d = 0; d < axis; d++) outer *= TensorShape(A)[d];
        Real *dA   = TensorGradients(A);
        Real *dout = TensorGradients(op->out);
        for (int o = 0; o < outer; o++) {
            for (int ix = 0; ix < idxSize; ix++) {
                int k = (int)(Ie[ix] + (Real)0.5);
                const Real *src = dout + ((size_t)o * idxSize  + ix) * innerSize;
                Real       *dst = dA   + ((size_t)o * axisSize + k)  * innerSize;
                int c = start;
                for (; c + VECTOR_DIMS <= end; c += VECTOR_DIMS)
                    VectorStore(dst + c, VectorAdd(VectorLoad(dst + c, FLOAT32), VectorLoad(src + c, FLOAT32)));
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
            outerOff = unravel(outer, TensorShape(A), TensorStride(A), axis);
            innerOff = unravel(inner, TensorShape(A) + axis + 1, TensorStride(A) + axis + 1,
                               TensorDimensions(A) - axis - 1);
            aStride  = TensorStride(A)[axis];
        }

        Real sum = 0;
        if (iC) {
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)(Ie[ix] + (Real)0.5) == kTarget)
                    sum += TensorGradients(op->out)[(outer * idxSize + ix) * innerSize + inner];
            }
        } else {
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = unravel(ix, TensorShape(I), TensorStride(I), TensorDimensions(I));
                if ((int)(Ie[idxOff] + (Real)0.5) == kTarget)
                    sum += TensorGradients(op->out)[(outer * idxSize + ix) * innerSize + inner];
            }
        }
        TensorGradients(A)[outerOff + kTarget * aStride + innerOff] += sum;
    }
}
