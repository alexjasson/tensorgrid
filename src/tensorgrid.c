// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "tensorgrid.h"
#include "utility.h"
#include "ThreadPool.h"
#include "DynamicArray.h"

// Kernel function arguments
typedef struct {
    Tensor out, A;
    Univariate op;
} UnaryArgs;
typedef struct {
    Tensor out, A, B;
    Bivariate op;
} BinaryArgs;

typedef enum { Map, Zip, Multiply, Broadcast, Reduce } OperationType;

// Tensor operation representation - used for autograd
typedef struct {
    OperationType type;
    Tensor out, A, B;
    union { Univariate u; Bivariate b; };
    int axis;      // axis needed for reduce operation
    Entry initial; // initial value needed for reduce operation
} Operation;

// Kernel functions
static void mapSubset(void *arg, int start, int end);
static void zipSubset(void *arg, int start, int end);
static void multiplySubset(void *arg, int start, int end);

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
    t.refcount  = malloc(sizeof(int));
    if (!t.entries || (autograd && !t.gradients) || !t.refcount) {
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
    free(A.refcount);
}


Tensor TensorSlice(Tensor A, int axis, int index)
{
    Tensor t;
    t.dimensions = A.dimensions - 1;
    for (int i = 0; i < axis; i++) {
        t.shape[i]   = A.shape[i];
        t.strides[i] = A.strides[i];
    }
    for (int i = axis; i < t.dimensions; i++) {
        t.shape[i]   = A.shape[i + 1];
        t.strides[i] = A.strides[i + 1];
    }
    t.entries   = A.entries   + index * A.strides[axis];
    t.gradients = A.gradients ? A.gradients + index * A.strides[axis] : NULL;
    t.refcount  = NULL;
    return t;
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

    int axisSize = A.shape[axis];
    int outSize  = TensorSize(out);
    for (int o = 0; o < outSize; o++) {
        int base = 0, remainder = o;
        for (int d = 0; d < A.dimensions; d++) {
            int idx    = remainder / out.strides[d];
            remainder -= idx * out.strides[d];
            base      += idx * A.strides[d];
        }
        Entry result = initial;
        for (int k = 0; k < axisSize; k++)
            result = f.f(result, A.entries[base + k * A.strides[axis]]);
        out.entries[o] = result;
    }

    if (out.gradients && A.gradients)
        operationArrayPush(operationArrayGet(), (Operation){ .type=Reduce, .out=out, .A=A, .axis=axis, .b=f, .initial=initial });
    return out;
}

Tensor TensorMap(Univariate op, Tensor A)
{
    Tensor out = TensorNew(A.dimensions, A.shape, A.gradients != NULL);
    UnaryArgs args = {out, A, op};
    threadPoolMap(threadPoolGet(), mapSubset, &args, TensorSize(A));

    if (out.gradients && A.gradients)
        operationArrayPush(operationArrayGet(),(Operation){ .type=Map, .out=out, .A=A, .u=op });
    return out;
}

Tensor TensorZip(Bivariate op, Tensor A, Tensor B)
{
    int autograd = A.gradients != NULL || B.gradients != NULL;
    Tensor out = TensorNew(A.dimensions, A.shape, autograd);
    BinaryArgs args = {out, A, B, op};
    threadPoolMap(threadPoolGet(), zipSubset, &args, TensorSize(A));

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
    threadPoolMap(threadPoolGet(), multiplySubset, &args, batchSize * m);

    if (out.gradients)
        operationArrayPush(operationArrayGet(),(Operation){ .type=Multiply, .out=out, .A=A, .B=B });
    return out;
}

Tensor TensorBroadcast(Tensor A, int n)
{
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < A.dimensions; i++) outShape[i] = A.shape[i];
    outShape[A.dimensions - 1] = n;
    Tensor out = TensorNew(A.dimensions, outShape, A.gradients != NULL);

    int batch = TensorSize(A);
    for (int i = 0; i < batch; i++)
        for (int j = 0; j < n; j++)
            out.entries[i * n + j] = A.entries[i];

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
    for (int i = 0; i < lossSize; i++)
        loss.gradients[i] = 1.0f / lossSize;

    // Accumulate gradients backwards
    for (int i = DynamicArraySize(arr) - 1; i >= 0; i--) {
        Operation *op = DynamicArrayGet(arr, i);

        switch (op->type) {
            case Map: {
                Entry *A    = op->A.entries;
                Entry *dA   = op->A.gradients;
                Entry *dout = op->out.gradients;
                int n       = TensorSize(op->out);

                if (dA) {
                    for (int j = 0; j < n; j++)
                        dA[j] += dout[j] * op->u.da(A[j]);
                }
                break;
            }

            case Zip: {
                Entry *A    = op->A.entries;
                Entry *B    = op->B.entries;
                Entry *dA   = op->A.gradients;
                Entry *dB   = op->B.gradients;
                Entry *dout = op->out.gradients;
                int n       = TensorSize(op->out);

                if (dA) {
                    for (int j = 0; j < n; j++)
                        dA[j] += dout[j] * op->b.da(A[j], B[j]);
                }

                if (dB) {
                    for (int j = 0; j < n; j++)
                        dB[j] += dout[j] * op->b.db(A[j], B[j]);
                }
                break;
            }

            case Multiply: { // out[m,n] = A[m,k] @ B[k,n]
                Entry *A    = op->A.entries;
                Entry *B    = op->B.entries;
                Entry *dA   = op->A.gradients;
                Entry *dB   = op->B.gradients;
                Entry *dout = op->out.gradients;

                int m = op->A.shape[op->A.dimensions - 2];
                int k = op->A.shape[op->A.dimensions - 1];
                int n = op->B.shape[op->B.dimensions - 1];

                if (dA) { // dA[m,k] += dout[m,n] @ B^T[n,k]
                    for (int row = 0; row < m; row++) {
                        for (int col = 0; col < k; col++) {
                            Entry sum = 0;
                            for (int j = 0; j < n; j++)
                                sum += dout[row * n + j] * B[col * n + j];
                            dA[row * k + col] += sum;
                        }
                    }
                }

                if (dB) { // dB[k,n] += A^T[k,m] @ dout[m,n]
                    for (int row = 0; row < k; row++) {
                        for (int col = 0; col < n; col++) {
                            Entry sum = 0;
                            for (int j = 0; j < m; j++)
                                sum += A[j * k + row] * dout[j * n + col];
                            dB[row * n + col] += sum;
                        }
                    }
                }
                break;
            }

            case Broadcast: { // out[rows,n] = broadcast(A[rows,1])
                Entry *dA   = op->A.gradients;
                Entry *dout = op->out.gradients;

                int n    = op->out.shape[op->out.dimensions - 1];
                int rows = TensorSize(op->out) / n;

                if (dA) { // dA[rows,1] += sum(dout[rows,n], axis=last)
                    for (int r = 0; r < rows; r++) {
                        Entry sum = 0;
                        for (int j = 0; j < n; j++)
                            sum += dout[r * n + j];
                        dA[r] += sum;
                    }
                }
                break;
            }

            case Reduce: { // out[...,1,...] = reduce(f, A[...,k,...], axis)
                Entry *dA   = op->A.gradients;
                Entry *dout = op->out.gradients;
                int outSize = TensorSize(op->out);

                if (dA) {
                    int axisSize   = op->A.shape[op->axis];
                    int axisStride = op->A.strides[op->axis];

                    Entry *result = malloc(axisSize * sizeof(Entry));
                    if (!result) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }

                    for (int o = 0; o < outSize; o++) {
                        int base = 0, remainder = o;
                        for (int d = 0; d < op->A.dimensions; d++) {
                            int idx = remainder / op->out.strides[d];
                            remainder    -= idx * op->out.strides[d];
                            base   += idx * op->A.strides[d];
                        }

                        result[0] = op->initial;
                        for (int k = 0; k < axisSize - 1; k++)
                            result[k + 1] = op->b.f(result[k], op->A.entries[base + k * axisStride]);

                        // dA[...,k,...] += dresult * db(result[...,k,...], A[...,k,...])
                        // dresult       *= da(result[...,k,...], A[...,k,...])
                        Entry dresult = dout[o];
                        for (int k = axisSize - 1; k >= 0; k--) {
                            Entry Ak = op->A.entries[base + k * axisStride];
                            dA[base + k * axisStride] += dresult * op->b.db(result[k], Ak);
                            dresult *= op->b.da(result[k], Ak);
                        }
                    }
                    free(result);
                }
                break;
            }
        }
    }
    operationArrayClear(arr);
}

void TensorSGD(Tensor A, float lr)
{
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        A.entries[i] -= lr * A.gradients[i];
}

Tensor TensorSoftmax(Tensor in, float temperature)
{
    // Numerical stability trick
    Tensor inView       = TensorView(in, in.dimensions, in.shape, 0);
    Tensor temp1        = TensorReduce(MAX, -INFINITY, inView, in.dimensions - 1);
    Tensor temp2        = TensorBroadcast(temp1, in.shape[in.dimensions - 1]);
    Tensor stableLogits = TensorZip(SUBTRACT, in, temp2);
    TensorFree(temp1); TensorFree(temp2);

    // Temperature scaling
    Tensor softmaxInput;
    if (temperature != 1.0f) {
        Tensor T = TensorNew(in.dimensions, in.shape, 0);
        int size = TensorSize(in);
        for (int i = 0; i < size; i++) T.entries[i] = temperature;
        Tensor scaledLogits = TensorZip(DIVIDE, stableLogits, T);
        TensorFree(stableLogits); TensorFree(T);
        softmaxInput = scaledLogits;
    } else {
        softmaxInput = stableLogits;
    }

    // out = exp(softmaxInput) ⊘ sum(exp(softmaxInput))
    Tensor A = TensorMap(EXP, softmaxInput);
    Tensor B = TensorReduce(ADD, 0.0f, A, in.dimensions - 1);
    Tensor C = TensorBroadcast(B, in.shape[in.dimensions - 1]);
    Tensor result = TensorZip(DIVIDE, A, C);

    TensorFree(softmaxInput);
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

int TensorSample(Tensor output, float temperature)
{
    Tensor P  = TensorSoftmax(output, temperature);
    float r   = randomFloat(0.0f, 1.0f);
    float cdf = 0.0f;
    int n     = TensorSize(P);
    for (int i = 0; i < n; i++) {
        cdf += P.entries[i];
        if (r <= cdf) { TensorFree(P); return i; }
    }
    TensorFree(P);
    return n - 1;
}

void TensorXavier(Tensor A)
{
    Entry bound = sqrtf(6.0f / (A.shape[A.dimensions - 2] + A.shape[A.dimensions - 1]));
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        A.entries[i] = (Entry)randomFloat(-bound, bound);
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

static void mapSubset(void *arg, int start, int end)
{
    UnaryArgs *args = arg;
    for (int i = start; i < end; i++)
        args->out.entries[i] = args->op.f(args->A.entries[i]);
}

static void zipSubset(void *arg, int start, int end)
{
    BinaryArgs *args = arg;
    for (int i = start; i < end; i++)
        args->out.entries[i] = args->op.f(args->A.entries[i], args->B.entries[i]);
}

static void multiplySubset(void *arg, int start, int end)
{
    BinaryArgs *args = arg;
    int m = args->A.shape[args->A.dimensions - 2];
    int k = args->A.shape[args->A.dimensions - 1];
    int n = args->B.shape[1];
    int strideA_row = args->A.strides[args->A.dimensions - 2];
    int strideA_col = args->A.strides[args->A.dimensions - 1];
    int strideB_row = args->B.strides[0];
    int strideB_col = args->B.strides[1];
    int batchStride = (args->A.dimensions >= 3) ? args->A.strides[args->A.dimensions - 3] : 0;

    for (int i = start; i < end; i++) {
        int batch = i / m;
        int row   = i % m;

        Entry *A_batch = args->A.entries + batch * batchStride;
        Entry *C_row   = args->out.entries + (batch * m + row) * n;

        for (int j = 0; j < n; j++)
            C_row[j] = 0.0f;

        for (int l = 0; l < k; l++) {
            Entry a = A_batch[row * strideA_row + l * strideA_col];
            for (int j = 0; j < n; j++)
                C_row[j] += a * args->B.entries[l * strideB_row + j * strideB_col];
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


