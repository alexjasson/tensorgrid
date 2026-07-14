// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tensorgrid.h"
#include "Operation.h"
#include "math/Scalar.h"
#include "adt/DynamicArray.h"
#include "adt/HashMap.h"
#include "adt/Stack.h"
#include "utility.h"
#include "adt/MemoryPool.h"
#include "math/Vector.h"
#include "math/Scalar.h"

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
    Real *m;     // SGD momentum or Adam first moment
    Real *v;     // Adam second moment
    int   t;     // Adam timestep
} optimizer;

struct tensorArray {
    DynamicArray tensors;
};

// Pointer to the entry at given index
static void *entry(Tensor A, size_t index) { return (unsigned char *)A.entries + index * ScalarSize(A.type); }

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

Tensor TensorNew(int dimensions, int *shape, DataType d, int autograd)
{
    if ((unsigned)d > REAL) {
        fprintf(stderr, "TensorNew requires a valid DataType!\n");
        exit(EXIT_FAILURE);
    }
    if ((d == FLOAT32 || d == FLOAT64) && ScalarSize(d) == sizeof(Real))
        d = REAL;
    if (autograd && d != REAL) {
        fprintf(stderr, "TensorNew autograd requires REAL entries!\n");
        exit(EXIT_FAILURE);
    }
    Tensor t = {0};
    t.dimensions = dimensions;
    t.type = d;
    t.scale = 1.0;
    for (int i = 0; i < dimensions; i++) t.shape[i] = shape[i];
    if (dimensions > 0) {
        t.strides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; i--)
            t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
    }
    int size = TensorSize(t);
    size_t footprint = (size_t)size * ScalarSize(d);
    MemoryPool pool = MemoryPoolGet();
    t.entries   = MemoryPoolAlloc(pool, footprint);
    memset(t.entries, 0, footprint);
    t.gradients = autograd ? MemoryPoolAlloc(pool, (size_t)size * sizeof(Real)) : NULL;
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

Tensor TensorArrayAdd(TensorArray ta, int dimensions, int *shape, int autograd)
{
    Tensor t = TensorNew(dimensions, shape, REAL, autograd);
    if (autograd) memset(t.gradients, 0, (size_t)TensorSize(t) * sizeof(Real));
    DynamicArrayPush(ta->tensors, &t);
    return t;
}

void TensorFree(Tensor A)
{
    if (!A.refcount || --(*A.refcount) > 0) return;
    size_t footprint = 1;
    for (int i = 0; i < A.dimensions; i++)
        if (A.strides[i] > 0) footprint += (size_t)(A.shape[i] - 1) * A.strides[i];
    MemoryPool pool = MemoryPoolGet();
    MemoryPoolFree(pool, A.entries, footprint * ScalarSize(A.type));
    MemoryPoolFree(pool, A.gradients, footprint * sizeof(Real));
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
    t.entries   = entry(A, (size_t)offset);
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
    t.entries   = entry(A, (size_t)offset);
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
    t.entries   = entry(A, (size_t)offset);
    t.gradients = A.gradients ? A.gradients + offset : NULL;
    t.refcount  = NULL;
    return t;
}

Tensor TensorReduce(Binary fn, Real initial, Tensor A, int axis)
{
    Tensor indices = TensorNew(1, (int[]){A.shape[axis]}, REAL, FALSE);
    Tensor out     = TensorScatter(fn, A, indices, initial, axis, 1);
    TensorFree(indices);
    return out;
}

Tensor TensorMap(Unary fn, Tensor A)                                      { return OperationMap(fn, A);                       }
Tensor TensorZip(Binary fn, Tensor A, Tensor B)                           { return OperationZip(fn, A, B);                    }
Tensor TensorMultiply(Tensor A, Tensor B)                                 { return OperationMultiply(A, B);                   }
Tensor TensorAffine(Unary fn, Tensor A, Tensor bias, int inner)           { return OperationAffine(fn, A, bias, inner);       }
Tensor TensorGather(Tensor A, Tensor indices, int axis)                   { return OperationGather(A, indices, axis);         }
Tensor TensorScatter(Binary fn, Tensor A, Tensor i, Real e, int a, int n) { return OperationScatter(fn, A, i, e, a, n);       }
Tensor TensorConvolution(Tensor in, Tensor filter, int s, int p)          { return OperationConvolution(in, filter, s, p);    }
Tensor TensorPool(Binary fn, Real e, Tensor in, int size, int s, int p)   { return OperationPool(fn, e, in, size, s, p);      }
Tensor TensorSoftmax(Tensor in, Real temperature)                         { return OperationSoftmax(in, temperature);         }
Tensor TensorCrossEntropyLoss(Tensor y, Tensor logits, Real temp)         { return OperationCrossEntropy(y, logits, temp);    }
Tensor TensorMSE(Tensor y, Tensor y_hat)                                  { return OperationMSE(y, y_hat);                    }
void TensorBackward(Tensor loss)                                          { OperationBackward(loss);                      }

Tensor TensorCopy(Tensor A)                                               { return TensorMap(IDENTITY, A);                }
Tensor TensorAdd(Tensor A, Tensor B)                                      { return TensorZip(ADD, A, B);                  }
Tensor TensorSubtract(Tensor A, Tensor B)                                 { return TensorZip(SUBTRACT, A, B);             }
Tensor TensorHadamard(Tensor A, Tensor B)                                 { return TensorZip(MULTIPLY, A, B);             }
Tensor TensorEmbedding(Tensor table, Tensor indices)                      { return TensorGather(table, indices, 0);       }
Tensor TensorMaxPool(Tensor A, int k, int s, int p)                       { return TensorPool(MAX, -INFINITE, A, k, s, p); }

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

Tensor TensorScale(Tensor A, Real scalar)
{
    if (!A.gradients && A.type != REAL) { A.scale *= scalar; return A; }
    if (!A.gradients) {
        Real *e = A.entries;
        int size = TensorSize(A);
        int contiguous = isContiguous(A.shape, A.strides, A.dimensions);
        for (int i = 0; i < size; i++) {
            int off = contiguous ? i : unravel(i, A.shape, A.strides, A.dimensions);
            e[off] *= scalar;
        }
        return A;
    }
    Tensor s = TensorNew(0, NULL, REAL, FALSE);
    TensorFill(s, scalar);
    Tensor out = TensorZip(MULTIPLY, A, s);
    TensorFree(s);
    TensorFree(A);
    return out;
}

void TensorFill(Tensor A, Real value)
{
    if (A.type != REAL || A.scale != 1.0) {
        fprintf(stderr, "TensorFill requires REAL entries, convert with TensorToType!\n");
        exit(EXIT_FAILURE);
    }
    Real *e = A.entries;
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        e[i] = value;
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
    Tensor indices = TensorNew(1, (int[]){n}, REAL, FALSE);
    Tensor out     = TensorGather(A, indices, axis);
    TensorFree(indices);
    return out;
}

Tensor TensorLayerNorm(Tensor in, Tensor weight, Tensor bias, Real eps)
{
    Tensor normalized = OperationNormalize(in, eps);
    Tensor scaled     = TensorZip(MULTIPLY, normalized, weight);
    Tensor result     = TensorZip(ADD, scaled, bias);
    TensorFree(normalized); TensorFree(scaled);
    return result;
}

Tensor TensorCausalMask(int n)
{
    Tensor mask = TensorNew(2, (int[]){n, n}, REAL, FALSE);
    Real *e = mask.entries;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            e[i * n + j] = (j <= i) ? 0.0f : -INFINITE;
    return mask;
}

Tensor TensorLogSoftmax(Tensor in, Real temperature)
{
    // Numerical stability trick
    Tensor inView       = TensorDetach(in);
    Tensor temp1        = TensorReduce(MAX, -INFINITE, inView, in.dimensions - 1);
    Tensor stableLogits = TensorZip(SUBTRACT, in, temp1);
    TensorFree(temp1);

    // Temperature scaling
    if (temperature != 1.0f)
        stableLogits = TensorScale(stableLogits, 1.0f / temperature);

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

    Tensor n = TensorNew(0, NULL, REAL, FALSE);
    TensorFill(n, (Real)N);
    Tensor result = TensorZip(DIVIDE, sum, n);

    TensorFree(A); TensorFree(B); TensorFree(sum); TensorFree(n);
    return result;
}

Tensor TensorSample(Tensor logits, Real temperature)
{
    if (logits.type != REAL || logits.scale != 1.0) {
        fprintf(stderr, "TensorSample requires REAL entries, convert with TensorToType!\n");
        exit(EXIT_FAILURE);
    }
    Real *le       = logits.entries;
    int numClasses = logits.shape[logits.dimensions - 1];
    int n          = TensorSize(logits) / numClasses;

    int outDims = logits.dimensions - 1;
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < outDims; i++) outShape[i] = logits.shape[i];
    Tensor out = TensorNew(outDims, outShape, REAL, 0);
    Real *oe   = out.entries;

    if (temperature == 0.0f) {
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int j = 1; j < numClasses; j++)
                if (le[i * numClasses + j] > le[i * numClasses + best]) best = j;
            oe[i] = (Real)best;
        }
    } else {
        Tensor P = TensorSoftmax(logits, temperature);
        Real *pe = P.entries;
        for (int i = 0; i < n; i++) {
            Real r   = randomNumber(0.0, 1.0);
            Real cdf = 0.0;
            oe[i] = (Real)(numClasses - 1);
            for (int j = 0; j < numClasses; j++) {
                cdf += pe[i * numClasses + j];
                if (r <= cdf) { oe[i] = (Real)j; break; }
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
        if (A.scale * ScalarLoad(entry(A, (size_t)aOff), A.type) != B.scale * ScalarLoad(entry(B, (size_t)bOff), B.type)) count++;
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

    Tensor out = TensorNew(A.dimensions, A.shape, A.type, 0);
    out.scale = A.scale;
    int rowSize = TensorSize(A) / n;
    if (isContiguous(A.shape, A.strides, A.dimensions)) {
        // Each row is contiguous, so permute rows with a single copy each
        for (int i = 0; i < n; i++)
            memcpy(entry(out, (size_t)i * rowSize),
                   entry(A, (size_t)perm[i] * rowSize),
                   (size_t)rowSize * ScalarSize(A.type));
    } else {
        int innerDims = A.dimensions - 1;
        for (int i = 0; i < n; i++) {
            int srcRowOff = perm[i] * A.strides[0];
            for (int j = 0; j < rowSize; j++) {
                int innerOff = unravel(j, A.shape + 1, A.strides + 1, innerDims);
                Real v = ScalarLoad(entry(A, (size_t)(srcRowOff + innerOff)), A.type);
                ScalarStore(entry(out, (size_t)i * rowSize + j), out.type, v);
            }
        }
    }
    free(perm);
    return out;
}

void TensorXavier(Tensor A)
{
    if (A.type != REAL || A.scale != 1.0) {
        fprintf(stderr, "TensorXavier requires REAL entries, convert with TensorToType!\n");
        exit(EXIT_FAILURE);
    }
    Real *e = A.entries;
    int receptive = 1;
    for (int i = 2; i < A.dimensions; i++) receptive *= A.shape[i];
    int fanSum = (A.shape[0] + (A.dimensions > 1 ? A.shape[1] : 0)) * receptive;
    Real bound = sqrte((Real)6.0 / fanSum);
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        e[i] = randomNumber(-bound, bound);
}

Tensor TensorOneHot(Tensor labels, int numClasses)
{
    int outShape[MAX_DIMENSIONS];
    for (int i = 0; i < labels.dimensions; i++) outShape[i] = labels.shape[i];
    outShape[labels.dimensions] = numClasses;

    Tensor out = TensorNew(labels.dimensions + 1, outShape, REAL, 0);
    Real *oe = out.entries;
    int numLabels = TensorSize(labels);

    for (int i = 0; i < numLabels; i++) {
        int off = unravel(i, labels.shape, labels.strides, labels.dimensions);
        oe[i * numClasses + (int)(labels.scale * ScalarLoad(entry(labels, (size_t)off), labels.type))] = 1.0f;
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
    TensorArray ta = TensorArrayNew();
    TensorArrayRead(ta, path);
    if (DynamicArraySize(ta->tensors) != 1) {
        fprintf(stderr, "The .tgd file does not contain a single Tensor: %s\n", path);
        TensorArrayFree(ta);
        exit(EXIT_FAILURE);
    }

    Tensor t = *(Tensor *)DynamicArrayGet(ta->tensors, 0);
    DynamicArrayFree(ta->tensors);
    free(ta);
    return t;
}

Tensor TensorReadData(char *path, int dimensions, int *shape, long offset, DataType d)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open file: %s\n", path); exit(EXIT_FAILURE); }
    if (offset < 0 || fseek(f, offset, SEEK_SET) != 0) {
        fprintf(stderr, "Cannot read file: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    Tensor t = TensorNew(dimensions, shape, d, FALSE);
    size_t data = (size_t)TensorSize(t) * ScalarSize(d);
    if (fread(t.entries, 1, data, f) != data) {
        fprintf(stderr, "Cannot read file: %s\n", path);
        TensorFree(t);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);
    return t;
}

static void toReal(void *arg, int start, int end)
{
    Operation *op = arg;
    const unsigned char *src = op->A.entries;
    Real *dst = op->out.entries;
    size_t esz = ScalarSize(op->A.type);
    Vector scale = VectorBcast(op->A.scale);
    int i = start;
    for (; i + VECTOR_DIMS <= end; i += VECTOR_DIMS)
        VectorStore(dst + i, VectorMul(VectorLoad(src + (size_t)i * esz, op->A.type), scale));
    for (; i < end; i++)
        dst[i] = op->A.scale * ScalarLoad(src + (size_t)i * esz, op->A.type);
}

Tensor TensorToType(Tensor A, DataType d)
{
    Tensor out = TensorNew(A.dimensions, A.shape, d, FALSE);
    int size = TensorSize(A);
    int contiguous = isContiguous(A.shape, A.strides, A.dimensions);

    if (out.type == REAL && contiguous) {
        Operation op = { .A = A, .out = out };
        OperationDispatch(toReal, &op, size, size);
        return out;
    }

    if (out.type <= INT32) { // For integer types, scale maps the largest value to the type's max
        Real typeMax = out.type == UINT8 ? UINT8_MAX : out.type == INT8 ? INT8_MAX : out.type == INT16 ? INT16_MAX : INT32_MAX;
        Real absmax  = 0.0;
        for (int i = 0; i < size; i++) {
            int off = contiguous ? i : unravel(i, A.shape, A.strides, A.dimensions);
            Real x = A.scale * ScalarLoad(entry(A, (size_t)off), A.type);
            if (x < 0.0) {
                if (out.type == UINT8) {
                    fprintf(stderr, "Negative values can't be mapped to UINT8!\n");
                    exit(EXIT_FAILURE);
                }
                x = -x;
            }
            if (x > absmax) absmax = x;
        }
        if (absmax > 0.0) out.scale = absmax / typeMax;
    }
    for (int i = 0; i < size; i++) {
        int off = contiguous ? i : unravel(i, A.shape, A.strides, A.dimensions);
        ScalarStore(entry(out, (size_t)i), out.type, A.scale * ScalarLoad(entry(A, (size_t)off), A.type) / out.scale);
    }
    return out;
}

void TensorWrite(Tensor A, char *path)
{
    TensorArray ta = TensorArrayNew();
    DynamicArrayPush(ta->tensors, &A);
    TensorArrayWrite(ta, path);
    DynamicArrayFree(ta->tensors);
    free(ta);
}

void TensorArrayWrite(TensorArray ta, char *path)
{
    // Open file
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open file: %s\n", path); exit(EXIT_FAILURE); }

    // Write header
    int n = DynamicArraySize(ta->tensors);
    uint32_t tensors = (uint32_t)n;

    if (fwrite("TGD", 1, 3, f) != 3 ||
        fwrite(&tensors, sizeof(tensors), 1, f) != 1) {
        fprintf(stderr, "Cannot write file: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    // Write each tensor
    for (int i = 0; i < n; i++) {
        Tensor *t = DynamicArrayGet(ta->tensors, i);
        if (!isContiguous(t->shape, t->strides, t->dimensions)) {
            fprintf(stderr, "Cannot write non-contiguous tensor: %s\n", path);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        // Write data type and scale, REAL is stored as its floating point width
        int32_t type = t->type == REAL ? (sizeof(Real) == 8 ? FLOAT64 : FLOAT32) : (int32_t)t->type;
        double scale = t->scale;
        if (fwrite(&type, sizeof(type), 1, f) != 1 ||
            fwrite(&scale, sizeof(scale), 1, f) != 1) {
            fprintf(stderr, "Cannot write file: %s\n", path);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        // Write tensor shape
        int32_t dimensions = t->dimensions;
        if (fwrite(&dimensions, sizeof(dimensions), 1, f) != 1) {
            fprintf(stderr, "Cannot write file: %s\n", path);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        for (int d = 0; d < t->dimensions; d++) {
            int32_t shape = t->shape[d];
            if (fwrite(&shape, sizeof(shape), 1, f) != 1) {
                fprintf(stderr, "Cannot write file: %s\n", path);
                fclose(f);
                exit(EXIT_FAILURE);
            }
        }

        // Write tensor entries
        size_t data = (size_t)TensorSize(*t) * ScalarSize(t->type);
        if (fwrite(t->entries, 1, data, f) != data) {
            fprintf(stderr, "Cannot write file: %s\n", path);
            fclose(f);
            exit(EXIT_FAILURE);
        }
    }

    fclose(f);
}

int TensorArrayRead(TensorArray ta, char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return FALSE;

    // Read and validate header
    char magic[3];
    uint32_t tensors;

    if (fread(magic, 1, 3, f) != 3 ||
        memcmp(magic, "TGD", 3) != 0 ||
        fread(&tensors, sizeof(tensors), 1, f) != 1 ||
        tensors > INT32_MAX) {
        fprintf(stderr, "Not a .tgd file: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    int existing = DynamicArraySize(ta->tensors);
    if (existing != 0 && (uint32_t)existing != tensors) {
        fprintf(stderr, "Tensor array does not match .tgd file: %s\n", path);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    int n = (int)tensors;
    for (int i = 0; i < n; i++) {
        // Read data type and scale
        int32_t type;
        double scale;
        if (fread(&type, sizeof(type), 1, f) != 1 || (uint32_t)type > FLOAT64 ||
            fread(&scale, sizeof(scale), 1, f) != 1)
            goto error;
        if ((type == FLOAT32 || type == FLOAT64) && ScalarSize(type) == sizeof(Real))
            type = REAL;

        // Read tensor shape
        int32_t dimensions;
        int shape[MAX_DIMENSIONS];

        if (fread(&dimensions, sizeof(dimensions), 1, f) != 1 ||
            dimensions < 0 || dimensions > MAX_DIMENSIONS)
            goto error;

        for (int d = 0; d < dimensions; d++) {
            int32_t dim;
            if (fread(&dim, sizeof(dim), 1, f) != 1 || dim <= 0)
                goto error;
            shape[d] = dim;
        }

        // Check if tensor already exists, otherwise create a new one
        Tensor *t;
        if (existing) {
            t = DynamicArrayGet(ta->tensors, i);
            int mismatch = dimensions != t->dimensions || type != (int32_t)t->type;
            for (int d = 0; !mismatch && d < dimensions; d++)
                mismatch = shape[d] != t->shape[d];
            if (mismatch) {
                fprintf(stderr, "Tensor array does not match .tgd file: %s\n", path);
                fclose(f);
                exit(EXIT_FAILURE);
            }
        } else {
            Tensor created = TensorNew(dimensions, shape, type, FALSE);
            DynamicArrayPush(ta->tensors, &created);
            t = DynamicArrayGet(ta->tensors, i);
        }
        t->scale = (Real)scale;

        // Read entries into the tensor
        size_t data = (size_t)TensorSize(*t) * ScalarSize(t->type);
        if (fread(t->entries, 1, data, f) != data)
            goto error;
    }

    fclose(f);
    return TRUE;

error:
    fprintf(stderr, "Cannot read .tgd file: %s\n", path);
    fclose(f);
    exit(EXIT_FAILURE);
}

Real TensorGet(Tensor A, int *indices)
{
    int idx = 0;
    for (int i = 0; i < A.dimensions; i++)
        idx += indices[i] * A.strides[i];
    return A.scale * ScalarLoad(entry(A, (size_t)idx), A.type);
}

void TensorSet(Tensor A, int *indices, Real value)
{
    int idx = 0;
    for (int i = 0; i < A.dimensions; i++)
        idx += indices[i] * A.strides[i];
    ScalarStore(entry(A, (size_t)idx), A.type, value / A.scale);
}

void TensorSGD(Tensor A, Real lr, Real momentum)
{
    if (!A.optimizer) return; // skip tensors without autograd
    Real *e = A.entries;
    int size = TensorSize(A);

    if (momentum == 0.0f) {
        for (int i = 0; i < size; i++)
            e[i] -= lr * A.gradients[i];
        return;
    }

    if (!A.optimizer->m) {
        A.optimizer->m = calloc(size, sizeof(Real));
        if (!A.optimizer->m) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    Real *m = A.optimizer->m;
    for (int i = 0; i < size; i++) {
        m[i] = momentum * m[i] + A.gradients[i];
        e[i] -= lr * m[i];
    }
}

void TensorArraySGD(TensorArray ta, Real lr, Real momentum)
{
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        TensorSGD(*(Tensor *)DynamicArrayGet(ta->tensors, i), lr, momentum);
}

void TensorAdam(Tensor A, Real lr, Real beta1, Real beta2, Real epsilon, Real weightDecay)
{
    if (!A.optimizer) return; // skip tensors without autograd
    Real *e = A.entries;
    int size = TensorSize(A);

    if (!A.optimizer->m) {
        A.optimizer->m = calloc(size, sizeof(Real));
        A.optimizer->v = calloc(size, sizeof(Real));
        if (!A.optimizer->m || !A.optimizer->v) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    A.optimizer->t++;
    Real *m = A.optimizer->m;
    Real *v = A.optimizer->v;
    Real biasM = 1.0 - powi(beta1, A.optimizer->t);
    Real biasV = 1.0 - powi(beta2, A.optimizer->t);
    for (int i = 0; i < size; i++) {
        Real g = A.gradients[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        Real mHat = m[i] / biasM;
        Real vHat = v[i] / biasV;
        e[i] -= lr * (mHat / (sqrte(vHat) + epsilon) + weightDecay * e[i]);
    }
}

void TensorArrayAdam(TensorArray ta, Real lr, Real beta1, Real beta2, Real epsilon, Real weightDecay)
{
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        TensorAdam(*(Tensor *)DynamicArrayGet(ta->tensors, i), lr, beta1, beta2, epsilon, weightDecay);
}

void TensorBPETrain(Tensor corpus, Tensor merges)
{
    int targetMerges = TensorShape(merges, 0);
    int n = TensorSize(corpus);
    int *ids = malloc((size_t)n * sizeof *ids);
    for (int i = 0; i < n; i++) ids[i] = (int)TensorGet(corpus, (int[MAX_DIMENSIONS]){i});

    // Each loop records most frequent adjacent pair into merges, then replaces that pair in ids with new token
    HashMap counts = HashMapNew(1024);
    for (int numMerges = 0; numMerges < targetMerges; numMerges++) {
        HashMapClear(counts);
        for (int i = 0; i < n - 1; i++) {
            uint64_t key = packi(ids[i], ids[i + 1]);
            long c = HashMapGet(counts, key);   // -1 if absent
            HashMapStore(counts, key, (c < 0 ? 0 : c) + 1);
        }

        uint64_t bestKey;
        if (!HashMapArgmax(counts, &bestKey)) break;  // No pairs left

        int a = (int)(bestKey >> 32);
        int b = (int)(bestKey & 0xffffffffu);
        TensorSet(merges, (int[MAX_DIMENSIONS]){numMerges, 0}, (Real)a);
        TensorSet(merges, (int[MAX_DIMENSIONS]){numMerges, 1}, (Real)b);

        int w = 0;
        for (int i = 0; i < n; ) {
            if (i < n - 1 && ids[i] == a && ids[i + 1] == b) { ids[w++] = 256 + numMerges; i += 2; }
            else ids[w++] = ids[i++];
        }
        n = w;
    }
    HashMapFree(counts);
    free(ids);
}

Tensor TensorBPEEncode(Tensor corpus, Tensor merges)
{
    int targetMerges = TensorShape(merges, 0);
    int n = TensorSize(corpus);
    int *ids = malloc((size_t)n * sizeof *ids);
    for (int i = 0; i < n; i++) ids[i] = (int)TensorGet(corpus, (int[MAX_DIMENSIONS]){i});
    HashMap rankMap = HashMapNew(1024);
    for (int r = 0; r < targetMerges; r++)
        HashMapStore(rankMap, packi((int)TensorGet(merges, (int[MAX_DIMENSIONS]){r, 0}), (int)TensorGet(merges, (int[MAX_DIMENSIONS]){r, 1})), r);

    // Greedily merges the pair that the merges tensor (tokenizer) learned earliest, until none are left
    while (TRUE) {
        long best = -1;
        for (int i = 0; i < n - 1; i++) {
            long r = HashMapGet(rankMap, packi(ids[i], ids[i + 1]));
            if (r >= 0 && (best < 0 || r < best)) best = r;
        }
        if (best < 0) break;

        int a = (int)TensorGet(merges, (int[MAX_DIMENSIONS]){(int)best, 0});
        int b = (int)TensorGet(merges, (int[MAX_DIMENSIONS]){(int)best, 1});
        int w = 0;
        for (int i = 0; i < n; ) {
            if (i < n - 1 && ids[i] == a && ids[i + 1] == b) { ids[w++] = 256 + (int)best; i += 2; }
            else ids[w++] = ids[i++];
        }
        n = w;
    }

    Tensor tokens = TensorNew(1, (int[]){n}, REAL, FALSE);
    for (int i = 0; i < n; i++) TensorSet(tokens, (int[MAX_DIMENSIONS]){i}, (Real)ids[i]);
    free(ids);
    HashMapFree(rankMap);
    return tokens;
}

Tensor TensorBPEDecode(Tensor tokens, Tensor merges)
{
    int n = TensorSize(tokens);
    int vocabSize = 256 + TensorShape(merges, 0);

    // Decode each token by walking the merge tree with a stack
    Stack stack = StackNew(sizeof(int));
    DynamicArray out = DynamicArrayNew(sizeof(unsigned char));
    for (int i = 0; i < n; i++) {
        int id = (int)TensorGet(tokens, (int[MAX_DIMENSIONS]){i});
        if (id < 0 || id >= vocabSize) continue;
        StackPush(stack, &id);
        int t;
        while (StackPop(stack, &t)) {
            if (t < 256) {
                unsigned char b = (unsigned char)t;
                DynamicArrayPush(out, &b);
            } else {
                int r = t - 256;
                int right = (int)TensorGet(merges, (int[MAX_DIMENSIONS]){r, 1});
                int left  = (int)TensorGet(merges, (int[MAX_DIMENSIONS]){r, 0});
                StackPush(stack, &right);
                StackPush(stack, &left);
            }
        }
    }
    StackFree(stack);

    int total = DynamicArraySize(out);
    Tensor bytes = TensorNew(1, (int[]){total}, REAL, FALSE);
    for (int i = 0; i < total; i++)
        TensorSet(bytes, (int[MAX_DIMENSIONS]){i}, (Real)*(unsigned char *)DynamicArrayGet(out, i));
    DynamicArrayFree(out);
    return bytes;
}
