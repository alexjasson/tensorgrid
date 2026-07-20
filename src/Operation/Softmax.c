// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include "../Operation.h"
#include "../math/Scalar.h"
#include "../utility.h"

typedef struct {
    int rows, classes;
    int logitsContiguous;
    Real invTemperature;
} SoftmaxContext;

static void OperationSoftmaxForward(Operation *op);
static void OperationSoftmaxBackward(Operation *op);
static void softmaxForward(void *arg, int start, int end);
static void softmaxBackward(void *arg, int start, int end);

Tensor OperationSoftmax(Tensor logits, Real temperature)
{
    if (temperature <= 0.0f) {
        fprintf(stderr, "OperationSoftmax requires positive temperature!\n");
        exit(EXIT_FAILURE);
    }
    SoftmaxContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->classes = TensorShape(logits)[TensorDimensions(logits) - 1];
    ctx->rows = TensorSize(logits) / ctx->classes;
    ctx->logitsContiguous = isContiguous(TensorShape(logits), TensorStride(logits), TensorDimensions(logits));
    ctx->invTemperature = 1.0f / temperature;

    Tensor out = TensorNew(TensorDimensions(logits), TensorShape(logits), REAL, TensorGradients(logits) != NULL);
    return OperationExecute(OperationNew(out, logits, NULL, OperationSoftmaxForward, OperationSoftmaxBackward, ctx));
}

static void OperationSoftmaxForward(Operation *op)
{
    SoftmaxContext *ctx = op->context;
    OperationDispatch(softmaxForward, op, ctx->rows, (long)ctx->rows * ctx->classes);
}

static void OperationSoftmaxBackward(Operation *op)
{
    if (!TensorGradients(op->A)) return;
    SoftmaxContext *ctx = op->context;
    OperationDispatch(softmaxBackward, op, ctx->rows, (long)ctx->rows * ctx->classes);
}

static void softmaxForward(void *arg, int start, int end)
{
    Operation *op = arg;
    SoftmaxContext *ctx = op->context;
    int classes = ctx->classes;
    Tensor L = op->A;
    Real *out = TensorEntries(op->out);

    for (int row = start; row < end; row++) {
        int xBase = ctx->logitsContiguous
                  ? row * classes
                  : unravel(row, TensorShape(L), TensorStride(L), TensorDimensions(L) - 1);
        int xStride = TensorStride(L)[TensorDimensions(L) - 1];
        Real *p = out + (size_t)row * classes;

        Real max = -INFINITE;
        for (int c = 0; c < classes; c++) {
            Real z = TensorGet(L, xBase + c * xStride) * ctx->invTemperature;
            if (z > max) max = z;
        }

        Real sum = 0.0f;
        for (int c = 0; c < classes; c++) {
            Real e = expe(TensorGet(L, xBase + c * xStride) * ctx->invTemperature - max);
            p[c] = e;
            sum += e;
        }
        Real inv = 1.0f / sum;
        for (int c = 0; c < classes; c++) p[c] *= inv;
    }
}

static void softmaxBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    SoftmaxContext *ctx = op->context;
    int classes = ctx->classes;
    Real *p = TensorEntries(op->out);
    Real *dp = TensorGradients(op->out);
    Real *dx = TensorGradients(op->A);

    for (int row = start; row < end; row++) {
        int xBase = ctx->logitsContiguous
                  ? row * classes
                  : unravel(row, TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A) - 1);
        int xStride = TensorStride(op->A)[TensorDimensions(op->A) - 1];
        Real *pr = p + (size_t)row * classes;
        Real *gr = dp + (size_t)row * classes;

        Real dot = 0.0f;
        for (int c = 0; c < classes; c++) dot += gr[c] * pr[c];
        for (int c = 0; c < classes; c++)
            dx[xBase + c * xStride] += ctx->invTemperature * pr[c] * (gr[c] - dot);
    }
}
