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

typedef enum { Map, Zip, Multiply, Broadcast, Reduce } OperationType;

// Tensor operation representation - used for autograd
typedef struct {
    OperationType type;
    Tensor out, A, B;
    union { Univariate u; Bivariate b; };
    int axis;      // axis needed for reduce operation
    Entry initial; // initial value needed for reduce operation
} Operation;

typedef struct optimizer {
    Entry *m;     // SGD momentum or Adam first moment
    Entry *v;     // Adam second moment
    int    t;     // Adam timestep
} optimizer;

// Kernel function arguments
typedef struct {
    Tensor out, A;
    Univariate op;
} UnaryArgs;
typedef struct {
    Tensor out, A, B;
    Bivariate op;
} BinaryArgs;
typedef struct {
    Tensor out, A;
    Bivariate op;
    int axis;
    Entry initial;
} ReduceArgs;
typedef struct {
    Tensor out, A;
    int n;
} BroadcastArgs;

// Kernel functions
static void mapForward(void *arg, int start, int end);
static void zipForward(void *arg, int start, int end);
static void multiplyForward(void *arg, int start, int end);
static void reduceForward(void *arg, int start, int end);
static void broadcastForward(void *arg, int start, int end);
static void mapBackward(void *arg, int start, int end);
static void zipBackward(void *arg, int start, int end);
static void multiplyBackwardDa(void *arg, int start, int end);
static void multiplyBackwardDb(void *arg, int start, int end);
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
    Tensor t;
    t.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++) t.shape[i] = shape[i];
    t.strides[dimensions - 1] = 1;
    for (int i = dimensions - 2; i >= 0; i--)
        t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
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
    A.strides[dimensions - 1] = 1;
    for (int i = dimensions - 2; i >= 0; i--)
        A.strides[i] = A.strides[i + 1] * A.shape[i + 1];
    return A;
}

Tensor TensorView(Tensor A, int dimensions, int *shape, int autograd)
{
    Tensor t;
    t.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++) t.shape[i] = shape[i];
    t.strides[dimensions - 1] = 1;
    for (int i = dimensions - 2; i >= 0; i--)
        t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
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
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < A.dimensions; i++) outShape[i] = A.shape[i];
    outShape[axis] = 1;
    Tensor out = TensorNew(A.dimensions, outShape, A.gradients != NULL);

    ReduceArgs args = {out, A, f, axis, initial};
    ThreadPoolMap(ThreadPoolGet(), reduceForward, &args, TensorSize(out));

    if (out.gradients && A.gradients)
        operationArrayPush(operationArrayGet(), (Operation){ .type=Reduce, .out=out, .A=A, .axis=axis, .b=f, .initial=initial });
    return out;
}

Tensor TensorMap(Univariate op, Tensor A)
{
    Tensor out = TensorNew(A.dimensions, A.shape, A.gradients != NULL);
    UnaryArgs args = {out, A, op};
    ThreadPoolMap(ThreadPoolGet(), mapForward, &args, TensorSize(A));

    if (out.gradients && A.gradients)
        operationArrayPush(operationArrayGet(),(Operation){ .type=Map, .out=out, .A=A, .u=op });
    return out;
}

Tensor TensorZip(Bivariate op, Tensor A, Tensor B)
{
    int autograd = A.gradients != NULL || B.gradients != NULL;
    Tensor out = TensorNew(A.dimensions, A.shape, autograd);
    BinaryArgs args = {out, A, B, op};
    ThreadPoolMap(ThreadPoolGet(), zipForward, &args, TensorSize(A));

    if (out.gradients)
        operationArrayPush(operationArrayGet(),(Operation){ .type=Zip, .out=out, .A=A, .B=B, .b=op });
    return out;
}

Tensor TensorMultiply(Tensor A, Tensor B)
{
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < A.dimensions; i++) outShape[i] = A.shape[i];
    outShape[A.dimensions - 1] = B.shape[B.dimensions - 1];
    int autograd = A.gradients != NULL || B.gradients != NULL;
    Tensor out = TensorNew(A.dimensions, outShape, autograd);

    int m         = A.shape[A.dimensions - 2];
    int k         = A.shape[A.dimensions - 1];
    int batchSize = TensorSize(A) / (m * k);
    BinaryArgs args = {out, A, B, {NULL, NULL, NULL}};
    ThreadPoolMap(ThreadPoolGet(), multiplyForward, &args, batchSize * m);

    if (out.gradients)
        operationArrayPush(operationArrayGet(),(Operation){ .type=Multiply, .out=out, .A=A, .B=B });
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

Tensor TensorBroadcast(Tensor A, int n)
{
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < A.dimensions; i++) outShape[i] = A.shape[i];
    outShape[A.dimensions - 1] = n;
    Tensor out = TensorNew(A.dimensions, outShape, A.gradients != NULL);

    BroadcastArgs args = {out, A, n};
    ThreadPoolMap(ThreadPoolGet(), broadcastForward, &args, TensorSize(A));

    if (out.gradients && A.gradients)
        operationArrayPush(operationArrayGet(),(Operation){ .type=Broadcast, .out=out, .A=A });
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
    TensorScale(loss, 1.0f / lossSize);

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

            case Broadcast: { // out[rows,n] = broadcast(A[rows,1])
                if (op->A.gradients) {
                    int n    = op->out.shape[op->out.dimensions - 1];
                    int rows = TensorSize(op->out) / n;
                    ThreadPoolMap(pool, broadcastBackward, op, rows);
                }
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
    float biasM = 1.0f - __builtin_powf(beta1, A.optimizer->t);
    float biasV = 1.0f - __builtin_powf(beta2, A.optimizer->t);
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
    Tensor temp2        = TensorBroadcast(temp1, in.shape[in.dimensions - 1]);
    Tensor stableLogits = TensorZip(SUBTRACT, in, temp2);
    TensorFree(temp1); TensorFree(temp2);

    // Temperature scaling
    if (temperature != 1.0f)
        TensorScale(stableLogits, 1.0f / temperature);

    // out = exp(stableLogits) ⊘ sum(exp(stableLogits))
    Tensor A = TensorMap(EXP, stableLogits);
    Tensor B = TensorReduce(ADD, 0.0f, A, in.dimensions - 1);
    Tensor C = TensorBroadcast(B, in.shape[in.dimensions - 1]);
    Tensor result = TensorZip(DIVIDE, A, C);

    TensorFree(stableLogits);
    TensorFree(A); TensorFree(B); TensorFree(C);
    return result;
}

Tensor TensorCrossEntropyLoss(Tensor y, Tensor y_hat)
{
    Tensor A = TensorMap(LOG, y_hat);
    Tensor B = TensorZip(MULTIPLY, y, A);
    Tensor C = TensorReduce(ADD, 0.0f, B, y.dimensions - 1);
    Tensor result = TensorMap(NEGATE, C);
    TensorFree(A); TensorFree(B); TensorFree(C);
    return result;
}

Tensor TensorMSE(Tensor y, Tensor y_hat)
{
    int numClasses = y.shape[y.dimensions - 1];
    Tensor A = TensorZip(SUBTRACT, y, y_hat);
    Tensor B = TensorZip(MULTIPLY, A, A);
    Tensor C = TensorReduce(ADD, 0.0f, B, y.dimensions - 1);
    Tensor n = TensorNew(C.dimensions, C.shape, FALSE);
    TensorFill(n, (Entry)numClasses);
    Tensor result = TensorZip(DIVIDE, C, n);
    TensorFree(A); TensorFree(B); TensorFree(C); TensorFree(n);
    return result;
}

Tensor TensorSample(Tensor logits, float temperature)
{
    int numClasses = logits.shape[logits.dimensions - 1];
    int n          = TensorSize(logits) / numClasses;

    int outDims    = logits.dimensions > 1 ? logits.dimensions - 1 : 1;
    int outShape[MAX_DIMENSIONS];
    if (logits.dimensions == 1) {
        outShape[0] = 1;
    } else {
        for (int i = 0; i < outDims; i++) outShape[i] = logits.shape[i];
    }
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
    Entry bound = __builtin_sqrtf(6.0f / (A.shape[A.dimensions - 2] + A.shape[A.dimensions - 1]));
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
    UnaryArgs *args = arg;
    for (int i = start; i < end; i++)
        args->out.entries[i] = args->op.f(args->A.entries[i]);
}

static void zipForward(void *arg, int start, int end)
{
    BinaryArgs *args = arg;
    for (int i = start; i < end; i++)
        args->out.entries[i] = args->op.f(args->A.entries[i], args->B.entries[i]);
}

static void multiplyForward(void *arg, int start, int end)
{
    BinaryArgs *args = arg;
    int m            = args->A.shape[args->A.dimensions - 2];
    int k            = args->A.shape[args->A.dimensions - 1];
    int n            = args->B.shape[args->B.dimensions - 1];
    int strideA_row  = args->A.strides[args->A.dimensions - 2];
    int strideA_col  = args->A.strides[args->A.dimensions - 1];
    int strideB_row  = args->B.strides[args->B.dimensions - 2];
    int strideB_col  = args->B.strides[args->B.dimensions - 1];
    int batchStrideA = (args->A.dimensions >= 3) ? args->A.strides[args->A.dimensions - 3] : 0;
    int batchStrideB = (args->B.dimensions >= 3) ? args->B.strides[args->B.dimensions - 3] : 0;

    for (int i = start; i < end; i++) {
        int batch = i / m;
        int row   = i % m;

        Entry *A_batch = args->A.entries + batch * batchStrideA;
        Entry *B_batch = args->B.entries + batch * batchStrideB;
        Entry *C_row   = args->out.entries + (batch * m + row) * n;

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
    ReduceArgs *args = arg;
    int axis     = args->axis;
    int axisSize = args->A.shape[axis];
    int aStride  = args->A.strides[axis];
    for (int o = start; o < end; o++) {
        int base = 0, remainder = o;
        for (int d = 0; d < args->A.dimensions; d++) {
            int idx    = remainder / args->out.strides[d];
            remainder -= idx * args->out.strides[d];
            base      += idx * args->A.strides[d];
        }
        Entry result = args->initial;
        for (int k = 0; k < axisSize; k++)
            result = args->op.f(result, args->A.entries[base + k * aStride]);
        args->out.entries[o] = result;
    }
}

static void broadcastForward(void *arg, int start, int end)
{
    BroadcastArgs *args = arg;
    int n = args->n;
    for (int i = start; i < end; i++)
        for (int j = 0; j < n; j++)
            args->out.entries[i * n + j] = args->A.entries[i];
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

static void broadcastBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *dA     = op->A.gradients;
    Entry *dout   = op->out.gradients;
    int n         = op->out.shape[op->out.dimensions - 1];
    for (int r = start; r < end; r++) {
        Entry sum = 0;
        for (int j = 0; j < n; j++)
            sum += dout[r * n + j];
        dA[r] += sum;
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
            base      += idx * op->A.strides[d];
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


