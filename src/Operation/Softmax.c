// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include "../Operation.h"
#include "../math/Vector.h"
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
    ctx->classes = logits.shape[logits.dimensions - 1];
    ctx->rows = TensorSize(logits) / ctx->classes;
    ctx->logitsContiguous = isContiguous(logits.shape, logits.strides, logits.dimensions);
    ctx->invTemperature = 1.0f / temperature;

    Tensor out = TensorNew(logits.dimensions, logits.shape, logits.gradients != NULL);
    return OperationExecute(OperationNew(out, logits, (Tensor){0}, OperationSoftmaxForward, OperationSoftmaxBackward, ctx));
}

static void OperationSoftmaxForward(Operation *op)
{
    SoftmaxContext *ctx = op->context;
    OperationDispatch(softmaxForward, op, ctx->rows, (long)ctx->rows * ctx->classes);
}

static void OperationSoftmaxBackward(Operation *op)
{
    if (!op->A.gradients) return;
    SoftmaxContext *ctx = op->context;
    OperationDispatch(softmaxBackward, op, ctx->rows, (long)ctx->rows * ctx->classes);
}

static void softmaxForward(void *arg, int start, int end)
{
    Operation *op = arg;
    SoftmaxContext *ctx = op->context;
    int classes = ctx->classes;
    Real *logits = op->A.entries;
    Real *out = op->out.entries;

    for (int row = start; row < end; row++) {
        int xBase = ctx->logitsContiguous
                  ? row * classes
                  : unravel(row, op->A.shape, op->A.strides, op->A.dimensions - 1);
        int xStride = op->A.strides[op->A.dimensions - 1];
        Real *p = out + (size_t)row * classes;

        Real max = -INFINITE;
        for (int c = 0; c < classes; c++) {
            Real z = logits[xBase + c * xStride] * ctx->invTemperature;
            if (z > max) max = z;
        }

        Real sum = 0.0f;
        for (int c = 0; c < classes; c++) {
            Real z = logits[xBase + c * xStride] * ctx->invTemperature - max;
            Real e = VectorFirst(expe(VectorBcast(z)));
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
    Real *p = op->out.entries;
    Real *dp = op->out.gradients;
    Real *dx = op->A.gradients;

    for (int row = start; row < end; row++) {
        int xBase = ctx->logitsContiguous
                  ? row * classes
                  : unravel(row, op->A.shape, op->A.strides, op->A.dimensions - 1);
        int xStride = op->A.strides[op->A.dimensions - 1];
        Real *pr = p + (size_t)row * classes;
        Real *gr = dp + (size_t)row * classes;

        Real dot = 0.0f;
        for (int c = 0; c < classes; c++) dot += gr[c] * pr[c];
        for (int c = 0; c < classes; c++)
            dx[xBase + c * xStride] += ctx->invTemperature * pr[c] * (gr[c] - dot);
    }
}
