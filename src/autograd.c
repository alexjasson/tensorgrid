// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "autograd.h"
#include "Operation.h"
#include "DynamicArray.h"

static DynamicArray operationArrayGet(void);
static void         operationArrayClear(DynamicArray arr);
static void         operationArrayPush(Operation op);

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
    }
    DynamicArrayClear(arr);
}

static void operationArrayPush(Operation op)
{
    DynamicArray arr = operationArrayGet();

    // Clears the array if the last operation is only referenced by the array
    // itself - meaning the user finished a forward pass without calling TensorBackward
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

void autogradBackward(Tensor loss)
{
    DynamicArray arr = operationArrayGet();

    // Zero all input gradients
    for (int i = 0; i < DynamicArraySize(arr); i++) {
        Operation *node = DynamicArrayGet(arr, i);
        if (node->A.gradients) memset(node->A.gradients, 0, TensorSize(node->A) * sizeof(Entry));
        if (node->B.gradients) memset(node->B.gradients, 0, TensorSize(node->B) * sizeof(Entry));
    }

    // Take mean over all loss entries
    int lossSize = TensorSize(loss);
    for (int i = 0; i < lossSize; i++)
        loss.gradients[i] = 1.0f / lossSize;

    // Accumulate gradients backwards
    for (int i = DynamicArraySize(arr) - 1; i >= 0; i--)
        OperationBackward(*(Operation *)DynamicArrayGet(arr, i));

    operationArrayClear(arr);
}

Tensor autogradMap(Univariate f, Tensor A)
{
    Tensor out = TensorNew(A.dimensions, A.shape, A.gradients != NULL);

    Operation op = { .type=Map, .out=out, .A=A, .u=f };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    return out;
}

Tensor autogradZip(Bivariate f, Tensor A, Tensor B)
{
    Tensor zipA = A;
    Tensor zipB = B;
    int freeA = FALSE;
    int freeB = FALSE;

    // Right-align shapes
    int outDims = A.dimensions >= B.dimensions ? A.dimensions : B.dimensions;
    Tensor *inputs[2] = { &zipA, &zipB };
    for (int n = 0; n < 2; n++) {
        Tensor *t = inputs[n];
        int pad = outDims - t->dimensions;
        for (int i = outDims - 1; i >= pad; i--) {
            t->shape[i]   = t->shape[i - pad];
            t->strides[i] = t->strides[i - pad];
        }
        for (int i = pad - 1; i >= 0; i--) {
            t->shape[i]   = 1;
            t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
        }
        t->dimensions = outDims;
    }

    int outShape[MAX_DIMENSIONS];
    for (int axis = 0; axis < outDims; axis++) {
        int aSize = zipA.shape[axis];
        int bSize = zipB.shape[axis];
        if (aSize != bSize && aSize != 1 && bSize != 1) {
            fprintf(stderr, "TensorZip shapes are not compatible!\n");
            exit(EXIT_FAILURE);
        }
        outShape[axis] = aSize >= bSize ? aSize : bSize;
    }

    for (int axis = 0; axis < outDims; axis++) {
        if (zipA.shape[axis] != outShape[axis]) {
            Tensor next = TensorBroadcast(zipA, axis, outShape[axis]);
            if (freeA) TensorFree(zipA);
            zipA = next;
            freeA = TRUE;
        }
        if (zipB.shape[axis] != outShape[axis]) {
            Tensor next = TensorBroadcast(zipB, axis, outShape[axis]);
            if (freeB) TensorFree(zipB);
            zipB = next;
            freeB = TRUE;
        }
    }

    int autograd = zipA.gradients != NULL || zipB.gradients != NULL;
    Tensor out = TensorNew(outDims, outShape, autograd);

    Operation op = { .type=Zip, .out=out, .A=zipA, .B=zipB, .b=f };
    OperationForward(op);

    if (op.out.gradients) operationArrayPush(op);
    if (freeA) TensorFree(zipA);
    if (freeB) TensorFree(zipB);
    return out;
}

Tensor autogradMultiply(Tensor A, Tensor B)
{
    int dimsA   = A.dimensions;
    int dimsB   = B.dimensions;
    int dimsOut = dimsA >= dimsB ? dimsA : dimsB;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < dimsOut - 2; i++)
        outShape[i] = (dimsA >= dimsB) ? A.shape[i] : B.shape[i];
    outShape[dimsOut - 2] = A.shape[dimsA - 2];
    outShape[dimsOut - 1] = B.shape[dimsB - 1];
    int autograd = A.gradients != NULL || B.gradients != NULL;
    Tensor out = TensorNew(dimsOut, outShape, autograd);

    Operation op = { .type=Multiply, .out=out, .A=A, .B=B };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    return out;
}

Tensor autogradAffine(Univariate f, Tensor product, Tensor bias, int inner)
{
    int autograd = product.gradients != NULL || bias.gradients != NULL;
    Tensor out = TensorNew(product.dimensions, product.shape, autograd);

    Operation op = { .type=Affine, .out=out, .A=product, .B=bias, .u=f, .inner=inner };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    return out;
}

Tensor autogradGather(Tensor A, Tensor B, int axis)
{
    int outDims = A.dimensions - 1 + B.dimensions;
    int outShape[MAX_DIMENSIONS];
    int p = 0;
    for (int d = 0; d < axis; d++) outShape[p++] = A.shape[d];
    for (int d = 0; d < B.dimensions; d++) outShape[p++] = B.shape[d];
    for (int d = axis + 1; d < A.dimensions; d++) outShape[p++] = A.shape[d];

    Tensor out = TensorNew(outDims, outShape, A.gradients != NULL);

    Operation op = { .type=Gather, .out=out, .A=A, .B=B, .axis=axis };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    return out;
}

Tensor autogradScatter(Bivariate f, Tensor A, Tensor B, Entry fill, int axis, int n)
{
    int outDims = A.dimensions - B.dimensions + 1;
    int outShape[MAX_DIMENSIONS];
    int p = 0;
    for (int d = 0; d < axis; d++) outShape[p++] = A.shape[d];
    outShape[p++] = n;
    for (int d = axis + B.dimensions; d < A.dimensions; d++) outShape[p++] = A.shape[d];

    Tensor out = TensorNew(outDims, outShape, A.gradients != NULL);

    Operation op = { .type=Scatter, .out=out, .A=A, .B=B, .axis=axis, .fill=fill, .b=f };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    return out;
}

Tensor autogradUnfold(Tensor A, Entry fill, int fH, int fW, int stride, int padding)
{
    int N    = A.shape[0];
    int Cin  = A.shape[1];
    int H    = A.shape[2];
    int W    = A.shape[3];
    int Hout = (H + 2 * padding - fH) / stride + 1;
    int Wout = (W + 2 * padding - fW) / stride + 1;
    int K    = Cin * fH * fW;
    int S    = Hout * Wout;

    Tensor windows = TensorNew(3, (int[]){N, K, S}, A.gradients != NULL);

    Operation op = { .type=Unfold, .out=windows, .A=A,
                     .stride=stride, .padding=padding, .fH=fH, .fW=fW, .fill=fill };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    return windows;
}
