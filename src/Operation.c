// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "Operation.h"
#include "adt/ThreadPool.h"
#include "adt/DynamicArray.h"

// Parallel threshold
#define PARALLEL (1 << 18)

// Helper functions
static void operationAdd(Operation *op, DynamicArray arr);
static void zeroGradients(Operation *op);

void OperationDispatch(ThreadPoolFn f, Operation *op, int total, long work)
{
    if (total <= 0) return;
    if (work < PARALLEL) f(op, 0, total);
    else ThreadPoolMap(ThreadPoolGet(), f, op, total);
}

// Adds the operations reachable from op in topological order
static void operationAdd(Operation *op, DynamicArray arr)
{
    if (!op || op->visited) return;
    op->visited = 1;
    if (op->A) operationAdd(OperationGet(op->A), arr);
    if (op->B) operationAdd(OperationGet(op->B), arr);
    DynamicArrayPush(arr, &op);
}

static void zeroGradients(Operation *op)
{
    Tensor inputs[] = { op->A, op->B };
    for (int n = 0; n < 2; n++) {
        Tensor input = inputs[n];
        if (!input) continue;
        Real *gradients = TensorGradients(input);
        if (!gradients) continue;
        size_t footprint = 1;
        for (int i = 0; i < TensorDimensions(input); i++)
            if (TensorStride(input)[i] > 0)
                footprint += (size_t)(TensorShape(input)[i] - 1) * TensorStride(input)[i];
        memset(gradients, 0, footprint * sizeof(Real));
    }
}

Operation OperationNew(Tensor out, Tensor A, Tensor B, OperationFn forward, OperationFn backward, void *context)
{
    Operation op = { .out = out, .A = A, .B = B,
                     .forward = forward,
                     .backward = backward,
                     .context = context };
    return op;
}

Tensor OperationExecute(Operation op)
{
    Tensor out = op.out, A = op.A, B = op.B;

    op.forward(&op);

    if (TensorGradients(op.out)) {
        if (A) op.A = TensorView(A, TensorDimensions(A), TensorShape(A), TensorStride(A));
        if (B) op.B = TensorView(B, TensorDimensions(B), TensorShape(B), TensorStride(B));
        Operation *node = malloc(sizeof(*node));
        if (!node) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
        *node = op;
        OperationSet(out, node);
    } else {
        free(op.context);
    }
    return out;
}

void OperationBackward(Tensor loss)
{
    DynamicArray arr = DynamicArrayNew(sizeof(Operation *));
    operationAdd(OperationGet(loss), arr);

    // Zero all input gradients
    for (int i = 0; i < DynamicArraySize(arr); i++)
        zeroGradients(*(Operation **)DynamicArrayGet(arr, i));

    // Take mean over all loss entries
    int lossSize = TensorSize(loss);
    for (int i = 0; i < lossSize; i++)
        TensorGradients(loss)[i] = 1.0f / lossSize;

    // Accumulate gradients backwards
    for (int i = DynamicArraySize(arr) - 1; i >= 0; i--) {
        Operation *op = *(Operation **)DynamicArrayGet(arr, i);
        op->backward(op);
        op->visited = 0;
    }
    DynamicArrayFree(arr);
}
