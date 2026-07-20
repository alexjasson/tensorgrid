// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include <stdlib.h>
#include "Operation.h"
#include "adt/ThreadPool.h"
#include "adt/DynamicArray.h"

// Parallel threshold
#define PARALLEL (1 << 18)

// The autograd tape
static DynamicArray operationArrayGet(void);
static void operationArrayClear(DynamicArray arr);
static void operationArrayPush(Operation op);

// Helper function
static void zeroGradients(Operation *op);

void OperationDispatch(ThreadPoolFn f, Operation *op, int total, long work)
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
        if (TensorReference(TensorOwner(last->out)) == 1)
            operationArrayClear(arr);
    }

    op.out = TensorView(op.out, TensorDimensions(op.out), TensorShape(op.out), TensorStride(op.out));
    if (op.A) op.A = TensorView(op.A, TensorDimensions(op.A), TensorShape(op.A), TensorStride(op.A));
    if (op.B) op.B = TensorView(op.B, TensorDimensions(op.B), TensorShape(op.B), TensorStride(op.B));
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

    // Cast entries to Real
    if (TensorGradients(out)) {
        if (A && (TensorDataType(A) != REAL || TensorScale(A))) {
            op.A = TensorCopy(A);
            TensorToType(op.A, REAL);
        }
        if (B && (TensorDataType(B) != REAL || TensorScale(B))) {
            op.B = TensorCopy(B);
            TensorToType(op.B, REAL);
        }
    }
    op.forward(&op);
    if (TensorGradients(op.out)) operationArrayPush(op);
    else free(op.context);
    if (op.A != A) TensorFree(op.A);
    if (op.B != B) TensorFree(op.B);
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
        TensorGradients(loss)[i] = 1.0f / lossSize;

    // Accumulate gradients backwards
    for (int i = DynamicArraySize(arr) - 1; i >= 0; i--) {
        Operation *op = DynamicArrayGet(arr, i);
        op->backward(op);
    }

    operationArrayClear(arr);
}

void OperationFlush(void)
{
    DynamicArray arr = operationArrayGet();
    int size = DynamicArraySize(arr);
    if (size > 0) {
        Operation *last = DynamicArrayGet(arr, size - 1);
        if (TensorReference(TensorOwner(last->out)) == 1)
            operationArrayClear(arr);
    }
}
