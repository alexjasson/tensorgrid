// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include "tensorgrid.h"
#include "utility.h"
#include "ThreadPool.h"
#include "DynamicArray.h"

// Univariate functions and derivatives
static Entry relu(Entry a)     { return a > 0.0f ? a : 0.0f; }
static Entry drelu(Entry a)    { return a > 0.0f ? 1.0f : 0.0f; }
static Entry sigmoid(Entry a)  { return 1.0f / (1.0f + fastExp(-a)); }
static Entry dsigmoid(Entry a) { return sigmoid(a) * (1.0f - sigmoid(a)); }
static Entry loge(Entry a)     { return fastLog(a); }
static Entry dloge(Entry a)    { return 1.0f / a; }
static Entry negate(Entry a)   { return -a; }
static Entry identity(Entry a) { return a; }
static Entry one(Entry a)      { (void)a; return  1.0f; }
static Entry minusOne(Entry a) { (void)a; return -1.0f; }
static Entry expe(Entry a)     { return fastExp(a); }

// Bivariate functions and partial derivatives
static Entry add(Entry a, Entry b)       { return a + b; }
static Entry subtract(Entry a, Entry b)  { return a - b; }
static Entry multiply(Entry a, Entry b)  { return a * b; }
static Entry divide(Entry a, Entry b)    { return a / b; }
static Entry max(Entry a, Entry b)       { return a > b ? a : b; }
static Entry first(Entry a, Entry b)     { (void)b; return a; }
static Entry second(Entry a, Entry b)    { (void)a; return b; }
static Entry ones(Entry a, Entry b)      { (void)a; (void)b; return  1.0f; }
static Entry minusOnes(Entry a, Entry b) { (void)a; (void)b; return -1.0f; }
static Entry divideDa(Entry a, Entry b)  { (void)a; return 1.0f / b; }
static Entry divideDb(Entry a, Entry b)  { return -a / (b * b); }
static Entry maxDa(Entry a, Entry b)     { return a >= b ? 1.0f : 0.0f; }
static Entry maxDb(Entry a, Entry b)     { return b > a ? 1.0f : 0.0f; }

Univariate RELU     = { relu,     drelu    };
Univariate SIGMOID  = { sigmoid,  dsigmoid };
Univariate LOG      = { loge,     dloge    };
Univariate EXP      = { expe,     expe     };
Univariate NEGATE   = { negate,   minusOne };
Univariate IDENTITY = { identity, one      };
Bivariate  ADD      = { add,      ones,      ones      };
Bivariate  SUBTRACT = { subtract, ones,      minusOnes };
Bivariate  MULTIPLY = { multiply, second,    first     };
Bivariate  DIVIDE   = { divide,   divideDa,  divideDb  };
Bivariate  MAX      = { max,      maxDa,     maxDb     };

typedef enum { Map, Zip, Multiply, Unfold, Broadcast, Reduce } OperationType;

// Tensor operation representation - used for autograd
typedef struct {
    OperationType type;
    Tensor out, A, B;
    union { Univariate u; Bivariate b; };
    int axis;      // axis needed for reduce operation
    Entry initial; // initial value needed for reduce operation
    int stride;    // stride needed for unfold operation
    int padding;   // padding needed for unfold operation
    int fH, fW;    // filter spatial dimensions needed for fold operation
} Operation;

typedef struct optimizer {
    Entry *m;     // SGD momentum or Adam first moment
    Entry *v;     // Adam second moment
    int    t;     // Adam timestep
} optimizer;

// Kernel functions
static void mapForward(void *arg, int start, int end);
static void zipForward(void *arg, int start, int end);
static void multiplyForward(void *arg, int start, int end);
static void unfold(void *arg, int start, int end);
static void reduceForward(void *arg, int start, int end);
static void broadcastForward(void *arg, int start, int end);
static void mapBackward(void *arg, int start, int end);
static void zipBackward(void *arg, int start, int end);
static void multiplyBackwardDa(void *arg, int start, int end);
static void multiplyBackwardDb(void *arg, int start, int end);
static void fold(void *arg, int start, int end);
static void broadcastBackward(void *arg, int start, int end);
static void reduceBackward(void *arg, int start, int end);

// Operation array functions
static DynamicArray operationArrayGet(void);
static void operationArrayPush(DynamicArray arr, Operation op);
static void operationArrayClear(DynamicArray arr);

int TensorSize(Tensor A)
{
    int size = 1;
    for (int i = 0; i < A.dimensions; i++)
        size *= A.shape[i];
    return size;
}

int TensorShape(Tensor A, int axis)
{
    return A.shape[axis];
}

Tensor TensorNew(int dimensions, int *shape, int autograd)
{
    Tensor t = {0};
    t.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++) t.shape[i] = shape[i];
    if (dimensions > 0) {
        t.strides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; i--)
            t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
    }
    int size    = TensorSize(t);
    t.entries   = calloc(size, sizeof(Entry));
    t.gradients = autograd ? calloc(size, sizeof(Entry)) : NULL;
    t.optimizer = autograd ? calloc(1, sizeof(optimizer)) : NULL;
    t.refcount  = malloc(sizeof(int));
    if (!t.entries || (autograd && (!t.gradients || !t.optimizer)) || !t.refcount) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    *t.refcount = 1;
    return t;
}

void TensorFree(Tensor A)
{
    if (!A.refcount || --(*A.refcount) > 0) return;
    free(A.entries);
    free(A.gradients);
    if (A.optimizer) {
        free(A.optimizer->m);
        free(A.optimizer->v);
        free(A.optimizer);
    }
    free(A.refcount);
}


Tensor TensorSlice(Tensor A, int axis, int index, int length)
{
    Tensor t         = A;
    t.shape[axis]    = length;
    t.entries        = A.entries   + index * A.strides[axis];
    t.gradients      = A.gradients ? A.gradients + index * A.strides[axis] : NULL;
    t.refcount       = NULL;
    return t;
}

Tensor TensorReshape(Tensor A, int dimensions, int *shape)
{
    A.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++) A.shape[i] = shape[i];
    if (dimensions > 0) {
        A.strides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; i--)
            A.strides[i] = A.strides[i + 1] * A.shape[i + 1];
    }
    return A;
}

Tensor TensorView(Tensor A, int dimensions, int *shape, int autograd)
{
    Tensor t = {0};
    t.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++) t.shape[i] = shape[i];
    if (dimensions > 0) {
        t.strides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; i--)
            t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
    }
    t.entries   = A.entries;
    t.gradients = autograd ? A.gradients : NULL;
    t.optimizer = autograd ? A.optimizer : NULL;
    t.refcount  = NULL;
    return t;
}

Tensor TensorTranspose(Tensor A)
{
    Tensor t = A;
    t.shape[A.dimensions - 2]   = A.shape[A.dimensions - 1];
    t.shape[A.dimensions - 1]   = A.shape[A.dimensions - 2];
    t.strides[A.dimensions - 2] = A.strides[A.dimensions - 1];
    t.strides[A.dimensions - 1] = A.strides[A.dimensions - 2];
    return t;
}

Tensor TensorReduce(Bivariate f, Entry initial, Tensor A, int axis)
{
    int outDims = A.dimensions;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < A.dimensions; i++) {
        outShape[i] = (i == axis) ? 1 : A.shape[i];
    }
    Tensor out = TensorNew(outDims, outShape, A.gradients != NULL);

    Operation op = { .type=Reduce, .out=out, .A=A, .axis=axis, .b=f, .initial=initial };
    ThreadPoolMap(ThreadPoolGet(), reduceForward, &op, TensorSize(out));

    if (out.gradients && A.gradients) operationArrayPush(operationArrayGet(), op);
    return out;
}

Tensor TensorMap(Univariate f, Tensor A)
{
    Tensor out = TensorNew(A.dimensions, A.shape, A.gradients != NULL);

    Operation op = { .type=Map, .out=out, .A=A, .u=f };
    ThreadPoolMap(ThreadPoolGet(), mapForward, &op, TensorSize(A));

    if (out.gradients && A.gradients) operationArrayPush(operationArrayGet(), op);
    return out;
}

Tensor TensorZip(Bivariate f, Tensor A, Tensor B)
{
    Tensor zipA = A;
    Tensor zipB = B;
    int freeA = FALSE;
    int freeB = FALSE;

    if (A.dimensions != B.dimensions) {
        fprintf(stderr, "TensorZip inputs must have the same number of dimensions!\n");
        exit(EXIT_FAILURE);
    }

    int outDims = A.dimensions;
    int outShape[MAX_DIMENSIONS];
    for (int axis = 0; axis < outDims; axis++) {
        int aSize = A.shape[axis];
        int bSize = B.shape[axis];
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
    ThreadPoolMap(ThreadPoolGet(), zipForward, &op, TensorSize(out));

    if (out.gradients) operationArrayPush(operationArrayGet(), op);
    if (freeA) TensorFree(zipA);
    if (freeB) TensorFree(zipB);
    return out;
}

Tensor TensorMultiply(Tensor A, Tensor B)
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

    int m         = A.shape[dimsA - 2];
    int n         = B.shape[dimsB - 1];
    int batchSize = TensorSize(out) / (m * n);
    Operation op  = { .type=Multiply, .out=out, .A=A, .B=B };
    ThreadPoolMap(ThreadPoolGet(), multiplyForward, &op, batchSize * m);

    if (out.gradients) operationArrayPush(operationArrayGet(), op);
    return out;
}

// windows[N, K, S] = unfold(input)
static Tensor tensorUnfold(Tensor input, int fH, int fW, int stride, int padding, Entry fill)
{
    int N    = input.shape[0];
    int Cin  = input.shape[1];
    int H    = input.shape[2];
    int W    = input.shape[3];
    int Hout = (H + 2 * padding - fH) / stride + 1;
    int Wout = (W + 2 * padding - fW) / stride + 1;
    int K    = Cin * fH * fW;
    int S    = Hout * Wout;

    Tensor windows = TensorNew(3, (int[]){N, K, S}, input.gradients != NULL);

    Operation op = { .type=Unfold, .out=windows, .A=input,
                     .stride=stride, .padding=padding, .fH=fH, .fW=fW, .initial=fill };
    ThreadPoolMap(ThreadPoolGet(), unfold, &op, N * K);

    if (windows.gradients) operationArrayPush(operationArrayGet(), op);
    return windows;
}

Tensor TensorConvolution(Tensor input, Tensor filter, int stride, int padding)
{
    int N    = input.shape[0];
    int H    = input.shape[2];
    int W    = input.shape[3];
    int Cout = filter.shape[0];
    int Cin  = filter.shape[1];
    int fH   = filter.shape[2];
    int fW   = filter.shape[3];
    int Hout = (H + 2 * padding - fH) / stride + 1;
    int Wout = (W + 2 * padding - fW) / stride + 1;
    int K    = Cin * fH * fW;

    Tensor windows     = tensorUnfold(input, fH, fW, stride, padding, 0.0f);
    Tensor matrixInput = TensorReshape(filter, 2, (int[]){Cout, K});
    Tensor output      = TensorMultiply(matrixInput, windows);  // [N, Cout, S]
    TensorFree(windows);

    return TensorReshape(output, 4, (int[]){N, Cout, Hout, Wout});
}

Tensor TensorPool(Bivariate f, Entry initial, Tensor input, int size, int stride, int padding)
{
    int N    = input.shape[0];
    int C    = input.shape[1];
    int H    = input.shape[2];
    int W    = input.shape[3];
    int Hout = (H + 2 * padding - size) / stride + 1;
    int Wout = (W + 2 * padding - size) / stride + 1;

    Tensor flat   = TensorReshape(input, 4, (int[]){N * C, 1, H, W});
    Tensor cols   = tensorUnfold(flat, size, size, stride, padding, initial);  // [N*C, size*size, Hout*Wout]
    Tensor pooled = TensorReduce(f, initial, cols, 1);                         // [N*C, 1, Hout*Wout]
    Tensor out    = TensorReshape(pooled, 4, (int[]){N, C, Hout, Wout});
    TensorFree(cols);
    return out;
}

void TensorScale(Tensor A, float scalar)
{
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        A.entries[i] *= scalar;
}

void TensorFill(Tensor A, Entry value)
{
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        A.entries[i] = value;
}

Tensor TensorBroadcast(Tensor A, int axis, int n)
{
    if (axis < 0 || axis >= A.dimensions) {
        fprintf(stderr, "TensorBroadcast axis out of range!\n");
        exit(EXIT_FAILURE);
    }
    if (A.shape[axis] != 1) {
        fprintf(stderr, "TensorBroadcast axis must have size 1!\n");
        exit(EXIT_FAILURE);
    }

    int outDims = A.dimensions;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < outDims; i++) outShape[i] = A.shape[i];
    outShape[axis] = n;
    Tensor out = TensorNew(outDims, outShape, A.gradients != NULL);

    Operation op = { .type=Broadcast, .out=out, .A=A, .axis=axis };
    ThreadPoolMap(ThreadPoolGet(), broadcastForward, &op, TensorSize(A));

    if (out.gradients && A.gradients) operationArrayPush(operationArrayGet(), op);
    return out;
}

void TensorBackward(Tensor loss)
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
    ThreadPool pool = ThreadPoolGet();
    for (int i = DynamicArraySize(arr) - 1; i >= 0; i--) {
        Operation *op = DynamicArrayGet(arr, i);

        switch (op->type) {
            case Map: {
                if (op->A.gradients)
                    ThreadPoolMap(pool, mapBackward, op, TensorSize(op->out));
                break;
            }

            case Zip: {
                if (op->A.gradients || op->B.gradients)
                    ThreadPoolMap(pool, zipBackward, op, TensorSize(op->out));
                break;
            }

            case Multiply: { // out[...,m,n] = A[...,m,k] @ B[...,k,n]
                int m            = op->A.shape[op->A.dimensions - 2];
                int k            = op->A.shape[op->A.dimensions - 1];
                int n            = op->B.shape[op->B.dimensions - 1];
                int batchSize    = TensorSize(op->out) / (m * n);
                int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
                int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;

                if (op->A.gradients) {
                    int total = batchStrideA ? batchSize * m : m;
                    ThreadPoolMap(pool, multiplyBackwardDa, op, total);
                }
                if (op->B.gradients) {
                    int total = batchStrideB ? batchSize * k : k;
                    ThreadPoolMap(pool, multiplyBackwardDb, op, total);
                }
                break;
            }

            case Unfold: { // out[N, K, S] = unfold(A[N, Cin, H, W])
                if (op->A.gradients)
                    ThreadPoolMap(pool, fold, op, TensorSize(op->A));
                break;
            }

            case Broadcast: { // out[..., n, ...] = broadcast(A[..., 1, ...], axis)
                if (op->A.gradients)
                    ThreadPoolMap(pool, broadcastBackward, op, TensorSize(op->A));
                break;
            }

            case Reduce: { // out[...,1,...] = reduce(f, A[...,k,...], axis)
                if (op->A.gradients)
                    ThreadPoolMap(pool, reduceBackward, op, TensorSize(op->out));
                break;
            }
        }
    }
    operationArrayClear(arr);
}

void TensorSGD(Tensor A, float lr, float momentum)
{
    int size = TensorSize(A);

    if (momentum == 0.0f) {
        for (int i = 0; i < size; i++)
            A.entries[i] -= lr * A.gradients[i];
        return;
    }

    if (!A.optimizer->m) {
        A.optimizer->m = calloc(size, sizeof(Entry));
        if (!A.optimizer->m) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    Entry *m = A.optimizer->m;
    for (int i = 0; i < size; i++) {
        m[i] = momentum * m[i] + A.gradients[i];
        A.entries[i] -= lr * m[i];
    }
}

void TensorAdam(Tensor A, float lr, float beta1, float beta2, float epsilon, float weightDecay)
{
    int size = TensorSize(A);

    if (!A.optimizer->m) {
        A.optimizer->m = calloc(size, sizeof(Entry));
        A.optimizer->v = calloc(size, sizeof(Entry));
        if (!A.optimizer->m || !A.optimizer->v) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    A.optimizer->t++;
    Entry *m = A.optimizer->m;
    Entry *v = A.optimizer->v;
    float biasM = 1.0f - fastPow(beta1, A.optimizer->t);
    float biasV = 1.0f - fastPow(beta2, A.optimizer->t);
    for (int i = 0; i < size; i++) {
        Entry g = A.gradients[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float mHat = m[i] / biasM;
        float vHat = v[i] / biasV;
        A.entries[i] -= lr * (mHat / (__builtin_sqrtf(vHat) + epsilon) + weightDecay * A.entries[i]);
    }
}

Tensor TensorSoftmax(Tensor in, float temperature)
{
    // Numerical stability trick
    Tensor inView       = TensorView(in, in.dimensions, in.shape, 0);
    Tensor temp1        = TensorReduce(MAX, -FLT_MAX, inView, in.dimensions - 1);
    Tensor temp2        = TensorBroadcast(temp1, in.dimensions - 1, in.shape[in.dimensions - 1]);
    Tensor stableLogits = TensorZip(SUBTRACT, in, temp2);
    TensorFree(temp1); TensorFree(temp2);

    // Temperature scaling
    if (temperature != 1.0f)
        TensorScale(stableLogits, 1.0f / temperature);

    // out = exp(stableLogits) ⊘ sum(exp(stableLogits))
    Tensor A = TensorMap(EXP, stableLogits);
    Tensor B = TensorReduce(ADD, 0.0f, A, in.dimensions - 1);
    Tensor C = TensorBroadcast(B, in.dimensions - 1, in.shape[in.dimensions - 1]);
    Tensor result = TensorZip(DIVIDE, A, C);

    TensorFree(stableLogits);
    TensorFree(A); TensorFree(B); TensorFree(C);
    return result;
}

Tensor TensorLogSoftmax(Tensor in, float temperature)
{
    // Numerical stability trick
    Tensor inView       = TensorView(in, in.dimensions, in.shape, 0);
    Tensor temp1        = TensorReduce(MAX, -FLT_MAX, inView, in.dimensions - 1);
    Tensor temp2        = TensorBroadcast(temp1, in.dimensions - 1, in.shape[in.dimensions - 1]);
    Tensor stableLogits = TensorZip(SUBTRACT, in, temp2);
    TensorFree(temp1); TensorFree(temp2);

    // Temperature scaling
    if (temperature != 1.0f)
        TensorScale(stableLogits, 1.0f / temperature);

    // out = stableLogits - log(sum(exp(stableLogits)))
    Tensor A = TensorMap(EXP, stableLogits);
    Tensor B = TensorReduce(ADD, 0.0f, A, in.dimensions - 1);
    Tensor C = TensorMap(LOG, B);
    Tensor D = TensorBroadcast(C, in.dimensions - 1, in.shape[in.dimensions - 1]);
    Tensor result = TensorZip(SUBTRACT, stableLogits, D);

    TensorFree(stableLogits);
    TensorFree(A); TensorFree(B); TensorFree(C); TensorFree(D);
    return result;
}

Tensor TensorNLL(Tensor y, Tensor logP)
{
    int N = TensorSize(y) / y.shape[y.dimensions - 1];

    Tensor A   = TensorZip(MULTIPLY, y, logP);
    Tensor B   = TensorReduce(ADD, 0.0f, A, y.dimensions - 1);
    Tensor sum = TensorMap(NEGATE, B);

    while (TensorSize(sum) > 1) {
        int axis = 0;
        while (sum.shape[axis] == 1) axis++;
        Tensor next = TensorReduce(ADD, 0.0f, sum, axis);
        TensorFree(sum);
        sum = next;
    }
    sum = TensorReshape(sum, 0, NULL);

    Tensor n = TensorNew(0, NULL, FALSE);
    TensorFill(n, (Entry)N);
    Tensor result = TensorZip(DIVIDE, sum, n);

    TensorFree(A); TensorFree(B); TensorFree(sum); TensorFree(n);
    return result;
}

Tensor TensorMSE(Tensor y, Tensor y_hat)
{
    int total = TensorSize(y);

    Tensor A   = TensorZip(SUBTRACT, y, y_hat);
    Tensor sum = TensorZip(MULTIPLY, A, A);

    while (TensorSize(sum) > 1) {
        int axis = 0;
        while (sum.shape[axis] == 1) axis++;
        Tensor next = TensorReduce(ADD, 0.0f, sum, axis);
        TensorFree(sum);
        sum = next;
    }
    sum = TensorReshape(sum, 0, NULL);

    Tensor n = TensorNew(0, NULL, FALSE);
    TensorFill(n, (Entry)total);
    Tensor result = TensorZip(DIVIDE, sum, n);

    TensorFree(A); TensorFree(sum); TensorFree(n);
    return result;
}

Tensor TensorSample(Tensor logits, float temperature)
{
    int numClasses = logits.shape[logits.dimensions - 1];
    int n          = TensorSize(logits) / numClasses;

    int outDims = logits.dimensions - 1;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < outDims; i++) outShape[i] = logits.shape[i];
    Tensor out = TensorNew(outDims, outShape, 0);

    if (temperature == 0.0f) {
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int j = 1; j < numClasses; j++)
                if (logits.entries[i * numClasses + j] > logits.entries[i * numClasses + best]) best = j;
            out.entries[i] = (float)best;
        }
    } else {
        Tensor P = TensorSoftmax(logits, temperature);
        for (int i = 0; i < n; i++) {
            float r   = randomFloat(0.0f, 1.0f);
            float cdf = 0.0f;
            out.entries[i] = (float)(numClasses - 1);
            for (int j = 0; j < numClasses; j++) {
                cdf += P.entries[i * numClasses + j];
                if (r <= cdf) { out.entries[i] = (float)j; break; }
            }
        }
        TensorFree(P);
    }
    return out;
}

int TensorHammingDistance(Tensor A, Tensor B)
{
    int n = TensorSize(A);
    int count = 0;
    for (int i = 0; i < n; i++)
        if (A.entries[i] != B.entries[i]) count++;
    return count;
}


Tensor TensorShuffle(Tensor A, unsigned int seed)
{
    int n = A.shape[0];
    int *perm = malloc(n * sizeof(int));
    if (!perm) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
    for (int i = 0; i < n; i++) perm[i] = i;
    for (int i = n - 1; i > 0; i--) {
        // xorshift
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        int j     = (int)(seed % (unsigned int)(i + 1));
        int tmp   = perm[i];
        perm[i]   = perm[j];
        perm[j]   = tmp;
    }

    Tensor out = TensorNew(A.dimensions, A.shape, 0);
    int rowSize = TensorSize(A) / n;
    for (int i = 0; i < n; i++) {
        memcpy(out.entries + i * rowSize,
               A.entries + perm[i] * rowSize,
               rowSize * sizeof(Entry));
    }
    free(perm);
    return out;
}

void TensorXavier(Tensor A)
{
    int receptive = 1;
    for (int i = 2; i < A.dimensions; i++) receptive *= A.shape[i];
    int fanSum = (A.shape[0] + (A.dimensions > 1 ? A.shape[1] : 0)) * receptive;
    Entry bound = __builtin_sqrtf(6.0f / fanSum);
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        A.entries[i] = (Entry)randomFloat(-bound, bound);
}

Tensor TensorOneHot(Tensor labels, int numClasses)
{
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < labels.dimensions; i++) outShape[i] = labels.shape[i];
    outShape[labels.dimensions] = numClasses;

    Tensor out = TensorNew(labels.dimensions + 1, outShape, 0);
    int numLabels = TensorSize(labels);

    for (int i = 0; i < numLabels; i++)
        out.entries[i * numClasses + (int)labels.entries[i]] = 1.0f;

    return out;
}


void TensorPrint(Tensor A)
{
    int size = TensorSize(A);
    int indices[MAX_DIMENSIONS] = {0};

    if (size == 1) {
        printf("%g\n", TensorGet(A, indices));
        return;
    }

    int prevClose = 0;
    for (int i = 0; i < size; i++) {
        int temp = i;
        for (int d = A.dimensions - 1; d >= 0; d--) {
            indices[d] = temp % A.shape[d];
            temp /= A.shape[d];
        }

        int open = 0;
        for (int d = A.dimensions - 1; d >= 0; d--) {
            if (indices[d] == 0) open++; else break;
        }

        int close = 0;
        for (int d = A.dimensions - 1; d >= 0; d--) {
            if (indices[d] == A.shape[d] - 1) close++; else break;
        }

        if (i > 0) {
            for (int b = 0; b < prevClose; b++) printf("]");
            printf(",");
        }
        for (int b = 0; b < open; b++) printf("[");
        printf("%g", TensorGet(A, indices));
        prevClose = close;
    }
    for (int b = 0; b < prevClose; b++) printf("]");
    printf("\n");
}

Tensor TensorRead(char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open file: %s\n", path); exit(EXIT_FAILURE); }

    // Open file and validate
    char magic[6];
    fread(magic, 1, 6, f);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) {
        fprintf(stderr, "Not a .npy file: %s\n", path);
        exit(EXIT_FAILURE);
    }

    // Read version number
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);
    (void)minor;

    // Read header
    uint32_t headerLength;
    if (major == 1) {
        uint16_t length;
        fread(&length, 2, 1, f);
        headerLength = length;
    } else {
        fread(&headerLength, 4, 1, f);
    }
    char *header = malloc(headerLength + 1);
    fread(header, 1, headerLength, f);
    header[headerLength] = '\0';

    if (strstr(header, "'fortran_order': True")) {
        fprintf(stderr, "Fortran-order arrays not supported\n");
        exit(EXIT_FAILURE);
    }

    // Parse data type
    char *start = strstr(header, "'descr'");
    start = strchr(start, ':') + 1;
    while (*start == ' ') start++;
    start++;
    char *end = strchr(start, '\'');
    char type[8] = {0};
    strncpy(type, start, end - start < 7 ? end - start : 7);
    char   dataType = type[1];
    size_t entrySize = (size_t)atoi(&type[2]);

    // Parse shape
    int shape[MAX_DIMENSIONS];
    int dimensions = 0;
    start = strchr(strstr(header, "'shape'"), '(') + 1;
    end   = strchr(start, ')');
    char *curr  = start;
    while (curr < end) {
        while (curr < end && (*curr == ' ' || *curr == ',')) curr++;
        if (curr >= end) break;
        int n;
        if (sscanf(curr, "%d", &n) != 1) break;
        shape[dimensions++] = n;
        while (*curr >= '0' && *curr <= '9') curr++;
    }

    free(header);

    int size = 1;
    for (int i = 0; i < dimensions; i++) size *= shape[i];

    void *data = malloc(size * entrySize);
    fread(data, entrySize, size, f);
    fclose(f);

    Tensor t = TensorNew(dimensions, shape, FALSE);
    if (dataType == 'f' && entrySize == 4) {
        memcpy(t.entries, data, size * sizeof(float));
    } else if (dataType == 'f' && entrySize == 8) {
        double *src = data;
        for (int i = 0; i < size; i++) t.entries[i] = (float)src[i];
    } else if (dataType == 'i' && entrySize == 4) {
        int32_t *src = data;
        for (int i = 0; i < size; i++) t.entries[i] = (float)src[i];
    } else if (dataType == 'i' && entrySize == 8) {
        int64_t *src = data;
        for (int i = 0; i < size; i++) t.entries[i] = (float)src[i];
    } else {
        uint8_t *src = data;
        for (int i = 0; i < size; i++) t.entries[i] = (float)src[i];
    }

    free(data);
    return t;
}

void TensorWrite(Tensor A, char *path)
{
    // Build shape string
    char shape[256] = "(";
    for (int i = 0; i < A.dimensions; i++) {
        char dim[32];
        snprintf(dim, sizeof(dim), "%d, ", A.shape[i]);
        strcat(shape, dim);
    }
    strcat(shape, ")");

    // Build header string
    char header[512];
    int unpaddedLength      = snprintf(header, sizeof(header),
        "{'descr': '<f4', 'fortran_order': False, 'shape': %s, }", shape);
    int headerLength = ((unpaddedLength + 1 + 10 + 63) / 64) * 64 - 10;
    for (int i = unpaddedLength; i < headerLength - 1; i++) header[i] = ' ';
    header[headerLength - 1] = '\n';

    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open file: %s\n", path); exit(EXIT_FAILURE); }

    uint16_t length = (uint16_t)headerLength;
    fwrite("\x93NUMPY", 1, 6, f);
    fwrite("\x01\x00",  1, 2, f);
    fwrite(&length,     2, 1, f);
    fwrite(header,      1, headerLength, f);

    int size = TensorSize(A);
    fwrite(A.entries, sizeof(Entry), size, f);
    fclose(f);
}

Entry TensorGet(Tensor A, int *indices)
{
    int idx = 0;
    for (int i = 0; i < A.dimensions; i++)
        idx += indices[i] * A.strides[i];
    return A.entries[idx];
}

void TensorSet(Tensor A, int *indices, Entry value)
{
    int idx = 0;
    for (int i = 0; i < A.dimensions; i++)
        idx += indices[i] * A.strides[i];
    A.entries[idx] = value;
}

static void mapForward(void *arg, int start, int end)
{
    Operation *op = arg;
    for (int i = start; i < end; i++)
        op->out.entries[i] = op->u.f(op->A.entries[i]);
}

static void zipForward(void *arg, int start, int end)
{
    Operation *op = arg;
    for (int i = start; i < end; i++)
        op->out.entries[i] = op->b.f(op->A.entries[i], op->B.entries[i]);
}

static void multiplyForward(void *arg, int start, int end)
{
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int strideA_row  = op->A.strides[op->A.dimensions - 2];
    int strideA_col  = op->A.strides[op->A.dimensions - 1];
    int strideB_row  = op->B.strides[op->B.dimensions - 2];
    int strideB_col  = op->B.strides[op->B.dimensions - 1];
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;

    for (int i = start; i < end; i++) {
        int batch = i / m;
        int row   = i % m;

        Entry *A_batch = op->A.entries + batch * batchStrideA;
        Entry *B_batch = op->B.entries + batch * batchStrideB;
        Entry *C_row   = op->out.entries + (batch * m + row) * n;

        for (int j = 0; j < n; j++)
            C_row[j] = 0.0f;

        for (int l = 0; l < k; l++) {
            Entry a = A_batch[row * strideA_row + l * strideA_col];
            for (int j = 0; j < n; j++)
                C_row[j] += a * B_batch[l * strideB_row + j * strideB_col];
        }
    }
}

static void reduceForward(void *arg, int start, int end)
{
    Operation *op = arg;
    int axis     = op->axis;
    int axisSize = op->A.shape[axis];
    int aStride  = op->A.strides[axis];
    for (int o = start; o < end; o++) {
        int base = 0, remainder = o;
        for (int d = 0; d < op->A.dimensions; d++) {
            int idx    = remainder / op->out.strides[d];
            remainder -= idx * op->out.strides[d];
            if (d != axis) base += idx * op->A.strides[d];
        }
        Entry result = op->initial;
        for (int k = 0; k < axisSize; k++)
            result = op->b.f(result, op->A.entries[base + k * aStride]);
        op->out.entries[o] = result;
    }
}

static void broadcastForward(void *arg, int start, int end)
{
    Operation *op = arg;
    int axis  = op->axis;
    int n     = op->out.shape[axis];
    int inner = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) inner *= op->out.shape[d];
    for (int i = start; i < end; i++) {
        int outerIdx = i / inner;
        int innerIdx = i % inner;
        Entry src  = op->A.entries[i];
        for (int j = 0; j < n; j++)
            op->out.entries[outerIdx * n * inner + j * inner + innerIdx] = src;
    }
}

static void mapBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *A    = op->A.entries;
    Entry *dA   = op->A.gradients;
    Entry *dout = op->out.gradients;
    for (int j = start; j < end; j++)
        dA[j] += dout[j] * op->u.da(A[j]);
}

static void zipBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *A    = op->A.entries;
    Entry *B    = op->B.entries;
    Entry *dA   = op->A.gradients;
    Entry *dB   = op->B.gradients;
    Entry *dout = op->out.gradients;
    if (dA && dB) {
        for (int j = start; j < end; j++) {
            Entry g = dout[j];
            dA[j] += g * op->b.da(A[j], B[j]);
            dB[j] += g * op->b.db(A[j], B[j]);
        }
    } else if (dA) {
        for (int j = start; j < end; j++)
            dA[j] += dout[j] * op->b.da(A[j], B[j]);
    } else if (dB) {
        for (int j = start; j < end; j++)
            dB[j] += dout[j] * op->b.db(A[j], B[j]);
    }
}

static void multiplyBackwardDa(void *arg, int start, int end)
{
    // dA[m,k] += dout[m,n] @ B^T[n,k]
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int batchSize    = TensorSize(op->out) / (m * n);
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;

    for (int i = start; i < end; i++) {
        int batchFrom, batchTo, row;
        if (batchStrideA) { batchFrom = i / m; batchTo = batchFrom + 1; row = i % m; }
        else              { batchFrom = 0;     batchTo = batchSize;     row = i;     }

        Entry *dA = op->A.gradients + batchFrom * batchStrideA;
        for (int col = 0; col < k; col++) {
            Entry sum = 0;
            for (int batch = batchFrom; batch < batchTo; batch++) {
                Entry *B    = op->B.entries    + batch * batchStrideB;
                Entry *dout = op->out.gradients + batch * m * n;
                for (int j = 0; j < n; j++)
                    sum += dout[row * n + j] * B[col * n + j];
            }
            dA[row * k + col] += sum;
        }
    }
}

static void multiplyBackwardDb(void *arg, int start, int end)
{
    // dB[k,n] += A^T[k,m] @ dout[m,n]
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int batchSize    = TensorSize(op->out) / (m * n);
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;

    for (int i = start; i < end; i++) {
        int batchFrom, batchTo, row;
        if (batchStrideB) { batchFrom = i / k; batchTo = batchFrom + 1; row = i % k; }
        else              { batchFrom = 0;     batchTo = batchSize;     row = i;     }

        Entry *dB = op->B.gradients + batchFrom * batchStrideB;
        for (int col = 0; col < n; col++) {
            Entry sum = 0;
            for (int batch = batchFrom; batch < batchTo; batch++) {
                Entry *A    = op->A.entries    + batch * batchStrideA;
                Entry *dout = op->out.gradients + batch * m * n;
                for (int j = 0; j < m; j++)
                    sum += A[j * k + row] * dout[j * n + col];
            }
            dB[row * n + col] += sum;
        }
    }
}

static void unfold(void *arg, int start, int end)
{
    Operation *op = arg;
    int Cin    = op->A.shape[1];
    int H      = op->A.shape[2];
    int W      = op->A.shape[3];
    int fH     = op->fH,     fW      = op->fW;
    int stride = op->stride, padding = op->padding;
    int Hout   = (H + 2 * padding - fH) / stride + 1;
    int Wout   = (W + 2 * padding - fW) / stride + 1;
    int S      = Hout * Wout;
    int K      = Cin * fH * fW;
    Entry        fill    = op->initial;
    const Entry *input   = op->A.entries;
    Entry       *windows = op->out.entries;

    for (int i = start; i < end; i++) {
        int n  = i / K;
        int k  = i % K;
        int ci = k / (fH * fW);
        int fh = (k / fW) % fH;
        int fw = k % fW;
        Entry       *windowRow = windows + (n * K + k) * S;
        const Entry *inPlane  = input + (n * Cin + ci) * H * W;

        for (int oh = 0; oh < Hout; oh++) {
            int ih = oh * stride - padding + fh;
            if (ih < 0 || ih >= H) {
                for (int ow = 0; ow < Wout; ow++)
                    windowRow[oh * Wout + ow] = fill;
            } else {
                const Entry *row = inPlane + ih * W;
                for (int ow = 0; ow < Wout; ow++) {
                    int iw = ow * stride - padding + fw;
                    windowRow[oh * Wout + ow] = (iw >= 0 && iw < W) ? row[iw] : fill;
                }
            }
        }
    }
}

static void fold(void *arg, int start, int end)
{
    Operation *op = arg;
    int Cin    = op->A.shape[1];
    int H      = op->A.shape[2];
    int W      = op->A.shape[3];
    int fH     = op->fH,     fW      = op->fW;
    int stride = op->stride, padding = op->padding;
    int Hout   = (H + 2 * padding - fH) / stride + 1;
    int Wout   = (W + 2 * padding - fW) / stride + 1;
    int S      = Hout * Wout;
    int K      = Cin * fH * fW;
    Entry       *dInput   = op->A.gradients;
    const Entry *dWindow = op->out.gradients;

    for (int i = start; i < end; i++) {
        int n  = i / (Cin * H * W);
        int ci = (i / (H * W)) % Cin;
        int ih = (i / W) % H;
        int iw = i % W;
        const Entry *dP_n = dWindow + n * K * S;

        Entry sum = 0;
        for (int fh = 0; fh < fH; fh++) {
            int numH = ih + padding - fh;
            if (numH < 0 || numH % stride) continue;
            int oh = numH / stride;
            if (oh >= Hout) continue;
            for (int fw = 0; fw < fW; fw++) {
                int numW = iw + padding - fw;
                if (numW < 0 || numW % stride) continue;
                int ow = numW / stride;
                if (ow >= Wout) continue;
                int k = (ci * fH + fh) * fW + fw;
                sum += dP_n[k * S + oh * Wout + ow];
            }
        }
        dInput[i] += sum;
    }
}

static void broadcastBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *dA     = op->A.gradients;
    Entry *dout   = op->out.gradients;
    int axis      = op->axis;
    int n         = op->out.shape[axis];
    int inner     = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) inner *= op->out.shape[d];
    for (int i = start; i < end; i++) {
        int outerIdx = i / inner;
        int innerIdx = i % inner;
        Entry acc  = 0;
        for (int j = 0; j < n; j++)
            acc += dout[outerIdx * n * inner + j * inner + innerIdx];
        dA[i] += acc;
    }
}

static void reduceBackward(void *arg, int start, int end)
{
    Operation *op  = arg;
    Entry *dA      = op->A.gradients;
    Entry *dout    = op->out.gradients;
    int axisSize   = op->A.shape[op->axis];
    int axisStride = op->A.strides[op->axis];
    Entry result[axisSize];

    for (int o = start; o < end; o++) {
        int base = 0, remainder = o;
        for (int d = 0; d < op->A.dimensions; d++) {
            int idx    = remainder / op->out.strides[d];
            remainder -= idx * op->out.strides[d];
            if (d != op->axis) base += idx * op->A.strides[d];
        }

        result[0] = op->initial;
        for (int k = 0; k < axisSize - 1; k++)
            result[k + 1] = op->b.f(result[k], op->A.entries[base + k * axisStride]);

        Entry dresult = dout[o];
        for (int k = axisSize - 1; k >= 0; k--) {
            Entry Ak = op->A.entries[base + k * axisStride];
            dA[base + k * axisStride] += dresult * op->b.db(result[k], Ak);
            dresult *= op->b.da(result[k], Ak);
        }
    }
}

static DynamicArray operationArrayGet(void)
{
    static DynamicArray arr = NULL;
    if (!arr) arr = DynamicArrayNew(sizeof(Operation));
    return arr;
}

static void operationArrayPush(DynamicArray arr, Operation op)
{
    // Clears the array if the last operation is only referenced by the operation array
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
