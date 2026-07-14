// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Scalar.h"

typedef struct {
    int size;
    int aContiguous, bContiguous;
} MSEContext;

static void OperationMSEForward(Operation *op);
static void OperationMSEBackward(Operation *op);
static void mseBackward(void *arg, int start, int end);

Tensor OperationMSE(Tensor y, Tensor y_hat)
{
    if (y.dimensions != y_hat.dimensions) {
        fprintf(stderr, "OperationMSE shapes are incompatible!\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < y.dimensions; i++) {
        if (y.shape[i] != y_hat.shape[i]) {
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
    ctx->aContiguous = isContiguous(y.shape, y.strides, y.dimensions);
    ctx->bContiguous = isContiguous(y_hat.shape, y_hat.strides, y_hat.dimensions);

    Tensor out = TensorNew(0, NULL, REAL, y.gradients != NULL || y_hat.gradients != NULL);
    return OperationExecute(OperationNew(out, y, y_hat, OperationMSEForward, OperationMSEBackward, ctx));
}

static void OperationMSEForward(Operation *op)
{
    MSEContext *ctx = op->context;
    Tensor Y = op->A, Y2 = op->B;
    const unsigned char *ye  = (const unsigned char *)Y.entries;
    const unsigned char *ye2 = (const unsigned char *)Y2.entries;
    size_t eY = ScalarSize(Y.type), eY2 = ScalarSize(Y2.type);
    Real sum = 0.0f;

    for (int i = 0; i < ctx->size; i++) {
        int aOff = ctx->aContiguous ? i : unravel(i, Y.shape, Y.strides, Y.dimensions);
        int bOff = ctx->bContiguous ? i : unravel(i, Y2.shape, Y2.strides, Y2.dimensions);
        Real d = Y.scale  * ScalarLoad(ye  + (size_t)aOff * eY,  Y.type)
               - Y2.scale * ScalarLoad(ye2 + (size_t)bOff * eY2, Y2.type);
        sum += d * d;
    }

    *(Real *)op->out.entries = sum / ctx->size;
}

static void OperationMSEBackward(Operation *op)
{
    MSEContext *ctx = op->context;
    if (!op->A.gradients && !op->B.gradients) return;
    OperationDispatch(mseBackward, op, ctx->size, ctx->size);
}

static void mseBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    MSEContext *ctx = op->context;
    Real *ye = op->A.entries, *ye2 = op->B.entries;
    Real scale = 2.0f * op->out.gradients[0] / ctx->size;

    for (int i = start; i < end; i++) {
        int aOff = ctx->aContiguous ? i : unravel(i, op->A.shape, op->A.strides, op->A.dimensions);
        int bOff = ctx->bContiguous ? i : unravel(i, op->B.shape, op->B.strides, op->B.dimensions);
        Real d = ye[aOff] - ye2[bOff];
        if (op->A.gradients) op->A.gradients[aOff] += scale * d;
        if (op->B.gradients) op->B.gradients[bOff] -= scale * d;
    }
}
