// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <stdint.h>
#include "tensorgrid.h"
#include "Operation.h"
#include "adt/DynamicArray.h"
#include "utility.h"
#include "adt/MemoryPool.h"
#include "Vector.h"
#include "Scalar.h"

Unary  RELU     = { (f)relu,     (f)drelu,    (f)reluv,     (f)dreluv     };
Unary  SIGMOID  = { (f)sigmoid,  (f)dsigmoid, (f)sigmoidv,  (f)dsigmoidv  };
Unary  TANH     = { (f)tanhe,    (f)dtanhe,   (f)tanhv,     (f)dtanhv     };
Unary  LOG      = { (f)loge,     (f)dloge,    (f)logv,      (f)dlogv      };
Unary  EXP      = { (f)expe,     (f)expe,     (f)expv,      (f)expv       };
Unary  NEGATE   = { (f)negate,   (f)minusOne, (f)negatev,   (f)dnegatev   };
Unary  IDENTITY = { (f)identity, (f)one,      (f)identityv, (f)didentityv };
Unary  SQRT     = { (f)sqrte,    (f)dsqrte,   (f)sqrtv,     (f)dsqrtv     };
Unary  GELU     = { (f)gelu,     (f)dgelu,    (f)geluv,     (f)dgeluv     };
Binary ADD      = { (f)add,      (f)ones,     (f)ones,      (f)addv,      (f)onesv,     (f)onesv      };
Binary SUBTRACT = { (f)subtract, (f)ones,     (f)minusOnes, (f)subtractv, (f)onesv,     (f)minusOnesv };
Binary MULTIPLY = { (f)multiply, (f)second,   (f)first,     (f)multiplyv, (f)secondv,   (f)firstv     };
Binary DIVIDE   = { (f)divide,   (f)divideDa, (f)divideDb,  (f)dividev,   (f)divideDav, (f)divideDbv  };
Binary MAX      = { (f)max,      (f)maxDa,    (f)maxDb,     (f)maxv,      (f)maxDav,    (f)maxDbv     };
Binary MIN      = { (f)min,      (f)minDa,    (f)minDb,     (f)minv,      (f)minDav,    (f)minDbv     };

typedef struct optimizer {
    Entry *m;     // SGD momentum or Adam first moment
    Entry *v;     // Adam second moment
    int    t;     // Adam timestep
} optimizer;

struct tensorArray {
    DynamicArray tensors;
};

int TensorSize(Tensor A)
{
    int size = 1;
    for (int i = 0; i < A.dimensions; i++)
        size *= A.shape[i];
    return size;
}

int TensorArraySize(TensorArray ta)
{
    int total = 0;
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        total += TensorSize(*(Tensor *)DynamicArrayGet(ta->tensors, i));
    return total;
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
    int size     = TensorSize(t);
    size_t bytes = (size_t)size * sizeof(Entry);
    MemoryPool pool = MemoryPoolGet();
    t.entries   = MemoryPoolAlloc(pool, bytes);
    memset(t.entries, 0, bytes);
    t.gradients = autograd ? MemoryPoolAlloc(pool, bytes) : NULL;
    t.optimizer = autograd ? calloc(1, sizeof(optimizer)) : NULL;
    t.refcount  = malloc(sizeof(int));
    if (!t.entries || (autograd && (!t.gradients || !t.optimizer)) || !t.refcount) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    *t.refcount = 1;
    return t;
}

TensorArray TensorArrayNew(void)
{
    TensorArray ta = malloc(sizeof(struct tensorArray));
    ta->tensors = DynamicArrayNew(sizeof(Tensor));
    return ta;
}

Tensor TensorArrayAdd(TensorArray ta, int dimensions, int *shape)
{
    Tensor t = TensorNew(dimensions, shape, TRUE);
    memset(t.gradients, 0, (size_t)TensorSize(t) * sizeof(Entry));
    DynamicArrayPush(ta->tensors, &t);
    return t;
}

void TensorFree(Tensor A)
{
    if (!A.refcount || --(*A.refcount) > 0) return;
    size_t footprint = 1;
    for (int i = 0; i < A.dimensions; i++)
        if (A.strides[i] > 0) footprint += (size_t)(A.shape[i] - 1) * A.strides[i];
    size_t bytes = footprint * sizeof(Entry);
    MemoryPool pool = MemoryPoolGet();
    MemoryPoolFree(pool, A.entries, bytes);
    MemoryPoolFree(pool, A.gradients, bytes);
    if (A.optimizer) {
        free(A.optimizer->m);
        free(A.optimizer->v);
        free(A.optimizer);
    }
    free(A.refcount);
}

void TensorArrayFree(TensorArray ta)
{
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        TensorFree(*(Tensor *)DynamicArrayGet(ta->tensors, i));
    DynamicArrayFree(ta->tensors);
    free(ta);
}

Tensor TensorSlice(Tensor A, int index, int length)
{
    Tensor t    = A;
    t.shape[0]  = length;
    int offset  = index * A.strides[0];
    t.entries   = A.entries   + offset;
    t.gradients = A.gradients ? A.gradients + offset : NULL;
    t.refcount  = NULL;
    return t;
}

Tensor TensorIndex(Tensor A, int index)
{
    Tensor t     = A;
    t.dimensions = A.dimensions - 1;
    for (int i = 0; i < t.dimensions; i++) {
        t.shape[i]   = A.shape[i + 1];
        t.strides[i] = A.strides[i + 1];
    }
    int offset  = index * A.strides[0];
    t.entries   = A.entries   + offset;
    t.gradients = A.gradients ? A.gradients + offset : NULL;
    t.refcount  = NULL;
    return t;
}

Tensor TensorReshape(Tensor A, int dimensions, int *shape)
{
    if (!isContiguous(A.shape, A.strides, A.dimensions)) {
        fprintf(stderr, "TensorReshape requires a contiguous input!\n");
        exit(EXIT_FAILURE);
    }
    A.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++) A.shape[i] = shape[i];
    if (dimensions > 0) {
        A.strides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; i--)
            A.strides[i] = A.strides[i + 1] * A.shape[i + 1];
    }
    return A;
}

Tensor TensorDetach(Tensor A)
{
    Tensor t    = A;
    t.gradients = NULL;
    t.optimizer = NULL;
    t.refcount  = NULL;
    return t;
}

Tensor TensorTranspose(Tensor A)
{
    int dims = A.dimensions;
    Tensor t = A;
    t.shape[dims - 2]   = A.shape[dims - 1];
    t.shape[dims - 1]   = A.shape[dims - 2];
    t.strides[dims - 2] = A.strides[dims - 1];
    t.strides[dims - 1] = A.strides[dims - 2];
    t.refcount = NULL;
    return t;
}

Tensor TensorRestride(Tensor A, int dimensions, int *shape, int *strides, int offset)
{
    Tensor t = A;
    t.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++) {
        t.shape[i]   = shape[i];
        t.strides[i] = strides[i];
    }
    t.entries   = A.entries   + offset;
    t.gradients = A.gradients ? A.gradients + offset : NULL;
    t.refcount  = NULL;
    return t;
}

Tensor TensorReduce(Binary fn, Entry initial, Tensor A, int axis)
{
    Tensor indices = TensorNew(1, (int[]){A.shape[axis]}, FALSE);
    Tensor out     = TensorScatter(fn, A, indices, initial, axis, 1);
    TensorFree(indices);
    return out;
}

Tensor TensorMap(Unary fn, Tensor A)                                      { return OperationMap(fn, A);                   }
Tensor TensorZip(Binary fn, Tensor A, Tensor B)                           { return OperationZip(fn, A, B);                }
Tensor TensorMultiply(Tensor A, Tensor B)                                 { return OperationMultiply(A, B);               }
Tensor TensorAffine(Unary fn, Tensor A, Tensor bias, int inner)           { return OperationAffine(fn, A, bias, inner);   }
Tensor TensorGather(Tensor A, Tensor indices, int axis)                   { return OperationGather(A, indices, axis);     }
Tensor TensorScatter(Binary fn, Tensor A, Tensor i, Entry e, int a, int n){ return OperationScatter(fn, A, i, e, a, n);   }
Tensor TensorConvolution(Tensor in, Tensor filter, int s, int p)          { return OperationConvolution(in, filter, s, p);}
Tensor TensorPool(Binary fn, Entry e, Tensor in, int size, int s, int p)  { return OperationPool(fn, e, in, size, s, p);  }
Tensor TensorSoftmax(Tensor in, float temperature)                        { return OperationSoftmax(in, temperature);     }
Tensor TensorCrossEntropyLoss(Tensor y, Tensor logits, float temp)        { return OperationCrossEntropy(y, logits, temp);}
void TensorBackward(Tensor loss)                                          { OperationBackward(loss);                      }

Tensor TensorCopy(Tensor A)                                               { return TensorMap(IDENTITY, A);                }
Tensor TensorAdd(Tensor A, Tensor B)                                      { return TensorZip(ADD, A, B);                  }
Tensor TensorSubtract(Tensor A, Tensor B)                                 { return TensorZip(SUBTRACT, A, B);             }
Tensor TensorHadamard(Tensor A, Tensor B)                                 { return TensorZip(MULTIPLY, A, B);             }
Tensor TensorEmbedding(Tensor table, Tensor indices)                      { return TensorGather(table, indices, 0);       }
Tensor TensorMaxPool(Tensor A, int k, int s, int p)                       { return TensorPool(MAX, -FLT_MAX, A, k, s, p); }

Tensor TensorLinearLayer(Tensor in, Tensor weight, Tensor bias, Unary fn)
{
    Tensor product = TensorMultiply(in, weight);
    Tensor out     = TensorAffine(fn, product, bias, 1);
    TensorFree(product);
    return out;
}

Tensor TensorConvolutionLayer(Tensor in, Tensor filter, Tensor bias, int stride, int padding, Unary fn)
{
    Tensor conv  = TensorConvolution(in, filter, stride, padding);
    int inner    = conv.shape[2] * conv.shape[3];
    Tensor out   = TensorAffine(fn, conv, bias, inner);
    TensorFree(conv);
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
    Tensor indices = TensorNew(1, (int[]){n}, FALSE);
    Tensor out     = TensorGather(A, indices, axis);
    TensorFree(indices);
    return out;
}

Tensor TensorLayerNorm(Tensor in, Tensor weight, Tensor bias, float eps)
{
    Tensor normalized = OperationNormalize(in, eps);
    Tensor scaled     = TensorZip(MULTIPLY, normalized, weight);
    Tensor result     = TensorZip(ADD, scaled, bias);
    TensorFree(normalized); TensorFree(scaled);
    return result;
}

Tensor TensorCausalMask(int n)
{
    Tensor mask = TensorNew(2, (int[]){n, n}, FALSE);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            mask.entries[i * n + j] = (j <= i) ? 0.0f : -INFINITE;
    return mask;
}

Tensor TensorLogSoftmax(Tensor in, float temperature)
{
    // Numerical stability trick
    Tensor inView       = TensorDetach(in);
    Tensor temp1        = TensorReduce(MAX, -FLT_MAX, inView, in.dimensions - 1);
    Tensor stableLogits = TensorZip(SUBTRACT, in, temp1);
    TensorFree(temp1);

    // Temperature scaling
    if (temperature != 1.0f)
        TensorScale(stableLogits, 1.0f / temperature);

    // out = stableLogits - log(sum(exp(stableLogits)))
    Tensor A = TensorMap(EXP, stableLogits);
    Tensor B = TensorReduce(ADD, 0.0f, A, in.dimensions - 1);
    Tensor C = TensorMap(LOG, B);
    Tensor result = TensorZip(SUBTRACT, stableLogits, C);

    TensorFree(stableLogits);
    TensorFree(A); TensorFree(B); TensorFree(C);
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
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int bC = isContiguous(B.shape, B.strides, B.dimensions);
    for (int i = 0; i < n; i++) {
        int aOff = aC ? i : unravel(i, A.shape, A.strides, A.dimensions);
        int bOff = bC ? i : unravel(i, B.shape, B.strides, B.dimensions);
        if (A.entries[aOff] != B.entries[bOff]) count++;
    }
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
    if (isContiguous(A.shape, A.strides, A.dimensions)) {
        // Each row is contiguous, so permute rows with a single copy each
        for (int i = 0; i < n; i++)
            memcpy(out.entries + (size_t)i * rowSize,
                   A.entries + (size_t)perm[i] * rowSize,
                   (size_t)rowSize * sizeof(Entry));
    } else {
        int innerDims = A.dimensions - 1;
        for (int i = 0; i < n; i++) {
            int srcRowOff = perm[i] * A.strides[0];
            for (int j = 0; j < rowSize; j++) {
                int innerOff = unravel(j, A.shape + 1, A.strides + 1, innerDims);
                out.entries[i * rowSize + j] = A.entries[srcRowOff + innerOff];
            }
        }
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

    for (int i = 0; i < numLabels; i++) {
        int off = unravel(i, labels.shape, labels.strides, labels.dimensions);
        out.entries[i * numClasses + (int)labels.entries[off]] = 1.0f;
    }

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

void TensorArraySGD(TensorArray ta, float lr, float momentum)
{
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        TensorSGD(*(Tensor *)DynamicArrayGet(ta->tensors, i), lr, momentum);
}

// base raised to a non-negative integer exponent
static float powi(float base, int exp)
{
    float result = 1.0f;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
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
    float biasM = 1.0f - powi(beta1, A.optimizer->t);
    float biasV = 1.0f - powi(beta2, A.optimizer->t);
    for (int i = 0; i < size; i++) {
        Entry g = A.gradients[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        float mHat = m[i] / biasM;
        float vHat = v[i] / biasV;
        A.entries[i] -= lr * (mHat / (__builtin_sqrtf(vHat) + epsilon) + weightDecay * A.entries[i]);
    }
}

void TensorArrayAdam(TensorArray ta, float lr, float beta1, float beta2, float epsilon, float weightDecay)
{
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        TensorAdam(*(Tensor *)DynamicArrayGet(ta->tensors, i), lr, beta1, beta2, epsilon, weightDecay);
}
