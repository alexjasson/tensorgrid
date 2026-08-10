// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include "../Operation.h"
#include "../math/Vector.h"
#include "../utility.h"

typedef struct {
    int size;
    int aContiguous, bContiguous;
} MSEContext;

static void OperationMSEForward(Operation *op);
static void OperationMSEBackward(Operation *op);
static void mseBackward(void *arg, int start, int end);

Tensor OperationMSE(Tensor y, Tensor y_hat)
{
    if (TensorDimensions(y) != TensorDimensions(y_hat)) {
        fprintf(stderr, "OperationMSE shapes are incompatible!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < TensorDimensions(y); i++) {
        if (TensorShape(y)[i] != TensorShape(y_hat)[i]) {
            fprintf(stderr, "OperationMSE shapes are incompatible!\n");
            exit(EXIT_FAILURE);
        }
    }

    MSEContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->size = TensorSize(y);
    ctx->aContiguous = isContiguous(TensorShape(y), TensorStride(y), TensorDimensions(y));
    ctx->bContiguous = isContiguous(TensorShape(y_hat), TensorStride(y_hat), TensorDimensions(y_hat));

    Tensor out = TensorNew(0, NULL, FLOAT32, TensorGradients(y) != NULL || TensorGradients(y_hat) != NULL);
    return OperationExecute(OperationNew(out, y, y_hat, OperationMSEForward, OperationMSEBackward, ctx));
}

static void OperationMSEForward(Operation *op)
{
    MSEContext *ctx = op->context;
    Tensor Y = op->A, Y2 = op->B;
    Real sum = 0.0f;

    for (int i = 0; i < ctx->size; i++) {
        int aOff = ctx->aContiguous ? i : unravel(i, TensorShape(Y), TensorStride(Y), TensorDimensions(Y));
        int bOff = ctx->bContiguous ? i : unravel(i, TensorShape(Y2), TensorStride(Y2), TensorDimensions(Y2));
        Real d = TensorGet(Y, aOff) - TensorGet(Y2, bOff);
        sum += d * d;
    }

    *(Real *)TensorEntries(op->out) = sum / ctx->size;
}

static void OperationMSEBackward(Operation *op)
{
    MSEContext *ctx = op->context;
    if (!TensorGradients(op->A) && !TensorGradients(op->B)) return;
    OperationDispatch(mseBackward, op, ctx->size, ctx->size);
}

static void mseBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    MSEContext *ctx = op->context;
    Real *ye = TensorEntries(op->A), *ye2 = TensorEntries(op->B);
    Real scale = 2.0f * TensorGradients(op->out)[0] / ctx->size;

    for (int i = start; i < end; i++) {
        int aOff = ctx->aContiguous ? i : unravel(i, TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A));
        int bOff = ctx->bContiguous ? i : unravel(i, TensorShape(op->B), TensorStride(op->B), TensorDimensions(op->B));
        Real d = ye[aOff] - ye2[bOff];
        if (TensorGradients(op->A)) TensorGradients(op->A)[aOff] += scale * d;
        if (TensorGradients(op->B)) TensorGradients(op->B)[bOff] -= scale * d;
    }
}
