// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
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
        TensorFree(op->bias);
        TensorFree(op->product);
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

    if (op.out.refcount)     (*op.out.refcount)++;
    if (op.A.refcount)       (*op.A.refcount)++;
    if (op.B.refcount)       (*op.B.refcount)++;
    if (op.bias.refcount)    (*op.bias.refcount)++;
    if (op.product.refcount) (*op.product.refcount)++;
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
        if (node->bias.gradients) memset(node->bias.gradients, 0, TensorSize(node->bias) * sizeof(Entry));
        if (node->product.gradients) memset(node->product.gradients, 0, TensorSize(node->product) * sizeof(Entry));
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

Tensor autogradMap(Unary fn, Tensor A)
{
    Tensor out = TensorNew(A.dimensions, A.shape, A.gradients != NULL);

    Operation op = { .type=Map, .out=out, .A=A, .u=fn };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    return out;
}

Tensor autogradZip(Binary fn, Tensor A, Tensor B)
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

    Operation op = { .type=Zip, .out=out, .A=zipA, .B=zipB, .b=fn };
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

Tensor autogradLinear(Tensor in, Tensor weight, Tensor bias, Unary fn)
{
    // Output shape matches in @ weight (broadcasting the leading batch dims)
    int dimsA   = in.dimensions;
    int dimsB   = weight.dimensions;
    int dimsOut = dimsA >= dimsB ? dimsA : dimsB;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < dimsOut - 2; i++)
        outShape[i] = (dimsA >= dimsB) ? in.shape[i] : weight.shape[i];
    outShape[dimsOut - 2] = in.shape[dimsA - 2];
    outShape[dimsOut - 1] = weight.shape[dimsB - 1];

    int autograd = in.gradients != NULL || weight.gradients != NULL || bias.gradients != NULL;
    Tensor product = TensorNew(dimsOut, outShape, autograd); // cached pre-activation in @ weight
    Tensor out     = TensorNew(dimsOut, outShape, autograd);

    Operation op = { .type=Linear, .out=out, .A=in, .B=weight,
                     .bias=bias, .product=product, .u=fn, .inner=1 };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    TensorFree(product); // tape (when recorded) holds the remaining reference
    return out;
}

Tensor autogradConvolution(Tensor in, Tensor filter, Tensor bias, int stride, int padding, Unary fn)
{
    int N    = in.shape[0];
    int H    = in.shape[2];
    int W    = in.shape[3];
    int Cout = filter.shape[0];
    int fH   = filter.shape[2];
    int fW   = filter.shape[3];
    int Hout = (H + 2 * padding - fH) / stride + 1;
    int Wout = (W + 2 * padding - fW) / stride + 1;
    int autograd = in.gradients != NULL || filter.gradients != NULL || bias.gradients != NULL;

    Tensor product = TensorNew(4, (int[]){N, Cout, Hout, Wout}, autograd);
    Tensor out     = TensorNew(4, (int[]){N, Cout, Hout, Wout}, autograd);

    Operation op = { .type=Convolution, .out=out, .A=in, .B=filter, .bias=bias,
                     .product=product, .u=fn, .stride=stride, .padding=padding,
                     .fH=fH, .fW=fW };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    TensorFree(product);
    return out;
}

Tensor autogradPool(Binary fn, Entry initial, Tensor in, int size, int stride, int padding)
{
    int N    = in.shape[0];
    int C    = in.shape[1];
    int H    = in.shape[2];
    int W    = in.shape[3];
    int Hout = (H + 2 * padding - size) / stride + 1;
    int Wout = (W + 2 * padding - size) / stride + 1;
    int autograd = in.gradients != NULL;
    // Pools (max/min) cache the winning index so backward is O(1) per
    // output. Other reductions replay the fold in backward.
    int cacheArg = autograd && (fn.f == MAX.f || fn.f == MIN.f);

    Tensor out     = TensorNew(4, (int[]){N, C, Hout, Wout}, autograd);
    Tensor indices = cacheArg ? TensorNew(4, (int[]){N, C, Hout, Wout}, FALSE) : (Tensor){0};

    Operation op = { .type=Pool, .out=out, .A=in, .b=fn, .product=indices,
                     .fH=size, .fW=size, .stride=stride, .padding=padding,
                     .fill=initial };
    OperationForward(op);
    if (op.out.gradients) operationArrayPush(op);
    if (indices.refcount) TensorFree(indices);
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

Tensor autogradScatter(Binary fn, Tensor A, Tensor B, Entry fill, int axis, int n)
{
    int outDims = A.dimensions - B.dimensions + 1;
    int outShape[MAX_DIMENSIONS];
    int p = 0;
    for (int d = 0; d < axis; d++) outShape[p++] = A.shape[d];
    outShape[p++] = n;
    for (int d = axis + B.dimensions; d < A.dimensions; d++) outShape[p++] = A.shape[d];

    Tensor out = TensorNew(outDims, outShape, A.gradients != NULL);

    Operation op = { .type=Scatter, .out=out, .A=A, .B=B, .axis=axis, .fill=fill, .b=fn };
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
