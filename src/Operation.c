// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include <stdlib.h>
#include "Operation.h"
#include "ThreadPool.h"
#include "DynamicArray.h"

// Parallel threshold
#define PARALLEL (1 << 18)

// The autograd tape
static DynamicArray operationArrayGet(void);
static void operationArrayClear(DynamicArray arr);
static void operationArrayPush(Operation op);

// Helper function
static void zeroGradients(Operation *op);

void OperationDispatch(KernelFn f, Operation *op, int total, long work)
{
    if (total <= 0) return;
    if (work < PARALLEL) f(op, 0, total);
    else ThreadPoolMap(ThreadPoolGet(), f, op, total);
}

static DynamicArray operationArrayGet(void)
{
    static DynamicArray arr = NULL;
    if (!arr) arr = DynamicArrayNew(sizeof(Operation));
    return arr;
}

static void operationArrayClear(DynamicArray arr)
{
    for (int i = 0; i < DynamicArraySize(arr); i++) {
        Operation *op = DynamicArrayGet(arr, i);
        TensorFree(op->out);
        TensorFree(op->A);
        TensorFree(op->B);
        free(op->context);
    }
    DynamicArrayClear(arr);
}

static void operationArrayPush(Operation op)
{
    DynamicArray arr = operationArrayGet();

    // Clears the array if the last operation is only referenced by the array
    // itself - meaning the user finished a forward pass without calling backward.
    int size = DynamicArraySize(arr);
    if (size > 0) {
        Operation *last = DynamicArrayGet(arr, size - 1);
        if (last->out.refcount && *last->out.refcount == 1)
            operationArrayClear(arr);
    }

    if (op.out.refcount) (*op.out.refcount)++;
    if (op.A.refcount)   (*op.A.refcount)++;
    if (op.B.refcount)   (*op.B.refcount)++;
    DynamicArrayPush(arr, &op);
}

static void zeroGradients(Operation *op)
{
    Tensor *inputs[] = { &op->A, &op->B };
    for (int n = 0; n < 2; n++) {
        Entry *gradients = inputs[n]->gradients;
        if (!gradients) continue;
        size_t footprint = 1;
        for (int i = 0; i < inputs[n]->dimensions; i++)
            if (inputs[n]->strides[i] > 0)
                footprint += (size_t)(inputs[n]->shape[i] - 1) * inputs[n]->strides[i];
        memset(gradients, 0, footprint * sizeof(Entry));
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
    Tensor out = op.out;
    op.forward(&op);
    if (op.out.gradients) operationArrayPush(op);
    else free(op.context);
    return out;
}

void OperationBackward(Tensor loss)
{
    DynamicArray arr = operationArrayGet();

    // Zero all input gradients
    for (int i = 0; i < DynamicArraySize(arr); i++)
        zeroGradients(DynamicArrayGet(arr, i));

    // Take mean over all loss entries
    int lossSize = TensorSize(loss);
    for (int i = 0; i < lossSize; i++)
        loss.gradients[i] = 1.0f / lossSize;

    // Accumulate gradients backwards
    for (int i = DynamicArraySize(arr) - 1; i >= 0; i--) {
        Operation *op = DynamicArrayGet(arr, i);
        op->backward(op);
    }

    operationArrayClear(arr);
}
