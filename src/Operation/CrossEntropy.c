// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include "../Operation.h"
#include "../math/Vector.h"
#include "../utility.h"

typedef struct {
    int rows, classes;
    int distribution, logitsContiguous, targetContiguous;
    Real invTemperature;
} CrossEntropyContext;

static void OperationCrossEntropyForward(Operation *op);
static void OperationCrossEntropyBackward(Operation *op);
static void crossEntropyBackward(void *arg, int start, int end);

Tensor OperationCrossEntropy(Tensor labels, Tensor logits, Real temperature)
{
    if (temperature <= 0.0f) {
        fprintf(stderr, "OperationCrossEntropy requires positive temperature!\n");
        exit(EXIT_FAILURE);
    }
    if (TensorDimensions(logits) < 1) {
        fprintf(stderr, "OperationCrossEntropy requires logits with a class axis!\n");
        exit(EXIT_FAILURE);
    }
    int classes = TensorShape(logits)[TensorDimensions(logits) - 1];
    int rows = TensorSize(logits) / classes;
    int distribution = TensorDimensions(labels) == TensorDimensions(logits);
    int integerLabels = TensorDimensions(labels) == TensorDimensions(logits) - 1;
    for (int i = 0; distribution && i < TensorDimensions(labels); i++)
        distribution = TensorShape(labels)[i] == TensorShape(logits)[i];
    for (int i = 0; integerLabels && i < TensorDimensions(labels); i++)
        integerLabels = TensorShape(labels)[i] == TensorShape(logits)[i];

    if (!distribution && !integerLabels) {
        fprintf(stderr, "OperationCrossEntropy label/logit shapes are incompatible!\n");
        exit(EXIT_FAILURE);
    }

    CrossEntropyContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->rows = rows;
    ctx->classes = classes;
    ctx->distribution = distribution;
    ctx->logitsContiguous = isContiguous(TensorShape(logits), TensorStride(logits), TensorDimensions(logits));
    ctx->targetContiguous = isContiguous(TensorShape(labels), TensorStride(labels), TensorDimensions(labels));
    ctx->invTemperature = 1.0f / temperature;

    Tensor out = TensorNew(0, NULL, FLOAT32, TensorGradients(logits) != NULL || TensorGradients(labels) != NULL);
    return OperationExecute(OperationNew(out, logits, labels, OperationCrossEntropyForward, OperationCrossEntropyBackward, ctx));
}

static void OperationCrossEntropyForward(Operation *op)
{
    CrossEntropyContext *ctx = op->context;
    Tensor Z = op->A, Y = op->B;
    Real loss = 0.0f;

    for (int row = 0; row < ctx->rows; row++) {
        int zBase = ctx->logitsContiguous
                  ? row * ctx->classes
                  : unravel(row, TensorShape(Z), TensorStride(Z), TensorDimensions(Z) - 1);
        int zStride = TensorStride(Z)[TensorDimensions(Z) - 1];

        Real max = -INFINITE;
        for (int c = 0; c < ctx->classes; c++) {
            Real v = TensorGet(Z, zBase + c * zStride) * ctx->invTemperature;
            if (v > max) max = v;
        }

        Real sum = 0.0f;
        for (int c = 0; c < ctx->classes; c++)
            sum += VectorFirst(expe(VectorBcast(TensorGet(Z, zBase + c * zStride) * ctx->invTemperature - max)));
        Real logDenom = max + VectorFirst(loge(VectorBcast(sum)));

        if (ctx->distribution) {
            int yBase = ctx->targetContiguous
                      ? row * ctx->classes
                      : unravel(row, TensorShape(Y), TensorStride(Y), TensorDimensions(Y) - 1);
            int yStride = TensorStride(Y)[TensorDimensions(Y) - 1];
            for (int c = 0; c < ctx->classes; c++) {
                Real z = TensorGet(Z, zBase + c * zStride);
                Real y = TensorGet(Y, yBase + c * yStride);
                loss -= y * (z * ctx->invTemperature - logDenom);
            }
        } else {
            int off = ctx->targetContiguous ? row : unravel(row, TensorShape(Y), TensorStride(Y), TensorDimensions(Y));
            int target = (int)(TensorGet(Y, off) + (Real)0.5);
            if (target < 0 || target >= ctx->classes) {
                fprintf(stderr, "Cross-entropy label out of range!\n");
                exit(EXIT_FAILURE);
            }
            loss += logDenom - TensorGet(Z, zBase + target * zStride) * ctx->invTemperature;
        }
    }

    *(Real *)TensorEntries(op->out) = loss / ctx->rows;
}

static void OperationCrossEntropyBackward(Operation *op)
{
    CrossEntropyContext *ctx = op->context;
    if (!TensorGradients(op->A) && !(ctx->distribution && TensorGradients(op->B))) return;
    OperationDispatch(crossEntropyBackward, op, ctx->rows, (long)ctx->rows * ctx->classes);
}

static void crossEntropyBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    CrossEntropyContext *ctx = op->context;
    Real *ze = TensorEntries(op->A), *ye = TensorEntries(op->B);
    Real logitScale = TensorGradients(op->out)[0] * ctx->invTemperature / ctx->rows;
    Real targetScale = TensorGradients(op->out)[0] / ctx->rows;

    for (int row = start; row < end; row++) {
        int zBase = ctx->logitsContiguous
                  ? row * ctx->classes
                  : unravel(row, TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A) - 1);
        int zStride = TensorStride(op->A)[TensorDimensions(op->A) - 1];

        Real max = -INFINITE;
        for (int c = 0; c < ctx->classes; c++) {
            Real v = ze[zBase + c * zStride] * ctx->invTemperature;
            if (v > max) max = v;
        }

        Real sum = 0.0f;
        for (int c = 0; c < ctx->classes; c++)
            sum += VectorFirst(expe(VectorBcast(ze[zBase + c * zStride] * ctx->invTemperature - max)));
        Real inv = 1.0f / sum;
        Real logDenom = max + VectorFirst(loge(VectorBcast(sum)));

        Real targetSum = 1.0f;
        int target = -1;
        int yBase = 0, yStride = 0;
        if (ctx->distribution) {
            yBase = ctx->targetContiguous
                  ? row * ctx->classes
                  : unravel(row, TensorShape(op->B), TensorStride(op->B), TensorDimensions(op->B) - 1);
            yStride = TensorStride(op->B)[TensorDimensions(op->B) - 1];
            targetSum = 0.0f;
            for (int c = 0; c < ctx->classes; c++)
                targetSum += ye[yBase + c * yStride];
        } else {
            int off = ctx->targetContiguous ? row : unravel(row, TensorShape(op->B), TensorStride(op->B), TensorDimensions(op->B));
            target = (int)(ye[off] + (Real)0.5);
        }
        for (int c = 0; c < ctx->classes; c++) {
            Real z = ze[zBase + c * zStride];
            Real p = VectorFirst(expe(VectorBcast(z * ctx->invTemperature - max))) * inv;
            Real y = c == target ? 1.0f : 0.0f;
            if (ctx->distribution) {
                y = ye[yBase + c * yStride];
            }
            if (TensorGradients(op->A)) TensorGradients(op->A)[zBase + c * zStride] += logitScale * (targetSum * p - y);
            if (ctx->distribution && TensorGradients(op->B))
                TensorGradients(op->B)[yBase + c * yStride] += -targetScale * (z * ctx->invTemperature - logDenom);
        }
    }
}
