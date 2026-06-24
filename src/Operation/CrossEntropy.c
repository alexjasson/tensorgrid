// https://github.com/alexjasson/tensorgrid

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include "../Operation.h"
#include "../Scalar.h"
#include "../utility.h"

typedef struct {
    int rows, classes;
    int distribution, logitsContiguous, targetContiguous;
    float invTemperature;
} CrossEntropyContext;

static void OperationCrossEntropyForward(Operation *op);
static void OperationCrossEntropyBackward(Operation *op);
static void crossEntropyBackward(void *arg, int start, int end);

Tensor OperationCrossEntropy(Tensor labels, Tensor logits, float temperature)
{
    if (temperature <= 0.0f) {
        fprintf(stderr, "OperationCrossEntropy requires positive temperature!\n");
        exit(EXIT_FAILURE);
    }
    if (logits.dimensions < 1) {
        fprintf(stderr, "OperationCrossEntropy requires logits with a class axis!\n");
        exit(EXIT_FAILURE);
    }
    int classes = logits.shape[logits.dimensions - 1];
    int rows = TensorSize(logits) / classes;
    int distribution = labels.dimensions == logits.dimensions;
    int integerLabels = labels.dimensions == logits.dimensions - 1;
    for (int i = 0; distribution && i < labels.dimensions; i++)
        distribution = labels.shape[i] == logits.shape[i];
    for (int i = 0; integerLabels && i < labels.dimensions; i++)
        integerLabels = labels.shape[i] == logits.shape[i];

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
    ctx->logitsContiguous = isContiguous(logits.shape, logits.strides, logits.dimensions);
    ctx->targetContiguous = isContiguous(labels.shape, labels.strides, labels.dimensions);
    ctx->invTemperature = 1.0f / temperature;

    Tensor out = TensorNew(0, NULL, logits.gradients != NULL || labels.gradients != NULL);
    return OperationExecute(OperationNew(out, logits, labels, OperationCrossEntropyForward, OperationCrossEntropyBackward, ctx));
}

static void OperationCrossEntropyForward(Operation *op)
{
    CrossEntropyContext *ctx = op->context;
    Entry loss = 0.0f;

    for (int row = 0; row < ctx->rows; row++) {
        int zBase = ctx->logitsContiguous
                  ? row * ctx->classes
                  : unravel(row, op->A.shape, op->A.strides, op->A.dimensions - 1);
        int zStride = op->A.strides[op->A.dimensions - 1];

        Entry max = -FLT_MAX;
        for (int c = 0; c < ctx->classes; c++) {
            Entry v = op->A.entries[zBase + c * zStride] * ctx->invTemperature;
            if (v > max) max = v;
        }

        Entry sum = 0.0f;
        for (int c = 0; c < ctx->classes; c++)
            sum += expe(op->A.entries[zBase + c * zStride] * ctx->invTemperature - max);
        Entry logDenom = max + loge(sum);

        if (ctx->distribution) {
            int yBase = ctx->targetContiguous
                      ? row * ctx->classes
                      : unravel(row, op->B.shape, op->B.strides, op->B.dimensions - 1);
            int yStride = op->B.strides[op->B.dimensions - 1];
            for (int c = 0; c < ctx->classes; c++) {
                Entry z = op->A.entries[zBase + c * zStride];
                Entry y = op->B.entries[yBase + c * yStride];
                loss -= y * (z * ctx->invTemperature - logDenom);
            }
        } else {
            int off = ctx->targetContiguous ? row : unravel(row, op->B.shape, op->B.strides, op->B.dimensions);
            int target = (int)op->B.entries[off];
            if (target < 0 || target >= ctx->classes) {
                fprintf(stderr, "Cross-entropy label out of range!\n");
                exit(EXIT_FAILURE);
            }
            loss += logDenom - op->A.entries[zBase + target * zStride] * ctx->invTemperature;
        }
    }

    op->out.entries[0] = loss / ctx->rows;
}

static void OperationCrossEntropyBackward(Operation *op)
{
    CrossEntropyContext *ctx = op->context;
    if (!op->A.gradients && !(ctx->distribution && op->B.gradients)) return;
    OperationDispatch(crossEntropyBackward, op, ctx->rows, (long)ctx->rows * ctx->classes);
}

static void crossEntropyBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    CrossEntropyContext *ctx = op->context;
    Entry logitScale = op->out.gradients[0] * ctx->invTemperature / ctx->rows;
    Entry targetScale = op->out.gradients[0] / ctx->rows;

    for (int row = start; row < end; row++) {
        int zBase = ctx->logitsContiguous
                  ? row * ctx->classes
                  : unravel(row, op->A.shape, op->A.strides, op->A.dimensions - 1);
        int zStride = op->A.strides[op->A.dimensions - 1];

        Entry max = -FLT_MAX;
        for (int c = 0; c < ctx->classes; c++) {
            Entry v = op->A.entries[zBase + c * zStride] * ctx->invTemperature;
            if (v > max) max = v;
        }

        Entry sum = 0.0f;
        for (int c = 0; c < ctx->classes; c++)
            sum += expe(op->A.entries[zBase + c * zStride] * ctx->invTemperature - max);
        Entry inv = 1.0f / sum;
        Entry logDenom = max + loge(sum);

        Entry targetSum = 1.0f;
        int target = -1;
        int yBase = 0, yStride = 0;
        if (ctx->distribution) {
            yBase = ctx->targetContiguous
                  ? row * ctx->classes
                  : unravel(row, op->B.shape, op->B.strides, op->B.dimensions - 1);
            yStride = op->B.strides[op->B.dimensions - 1];
            targetSum = 0.0f;
            for (int c = 0; c < ctx->classes; c++)
                targetSum += op->B.entries[yBase + c * yStride];
        } else {
            int off = ctx->targetContiguous ? row : unravel(row, op->B.shape, op->B.strides, op->B.dimensions);
            target = (int)op->B.entries[off];
        }
        for (int c = 0; c < ctx->classes; c++) {
            Entry z = op->A.entries[zBase + c * zStride];
            Entry p = expe(z * ctx->invTemperature - max) * inv;
            Entry y = c == target ? 1.0f : 0.0f;
            if (ctx->distribution) {
                y = op->B.entries[yBase + c * yStride];
            }
            if (op->A.gradients) op->A.gradients[zBase + c * zStride] += logitScale * (targetSum * p - y);
            if (ctx->distribution && op->B.gradients)
                op->B.gradients[yBase + c * yStride] += -targetScale * (z * ctx->invTemperature - logDenom);
        }
    }
}
