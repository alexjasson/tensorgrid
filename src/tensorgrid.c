// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "tensorgrid.h"
#include "math/Scale.h"
#include "Operation.h"
#include "adt/DynamicArray.h"
#include "adt/HashMap.h"
#include "adt/Stack.h"
#include "utility.h"
#include "adt/MemoryPool.h"
#include "math/Vector.h"

Unary  RELU     = { (f)relu,     (f)drelu     };
Unary  SIGMOID  = { (f)sigmoid,  (f)dsigmoid  };
Unary  TANH     = { (f)tanhe,    (f)dtanhe    };
Unary  LOG      = { (f)loge,     (f)dloge     };
Unary  EXP      = { (f)expe,     (f)expe      };
Unary  NEGATE   = { (f)negate,   (f)minusOne  };
Unary  IDENTITY = { (f)identity, (f)one       };
Unary  SQRT     = { (f)sqrte,    (f)dsqrte    };
Unary  GELU     = { (f)gelu,     (f)dgelu     };
Binary ADD      = { (g)add,      (g)ones,     (g)ones      };
Binary SUBTRACT = { (g)subtract, (g)ones,     (g)minusOnes };
Binary MULTIPLY = { (g)multiply, (g)second,   (g)first     };
Binary DIVIDE   = { (g)divide,   (g)divideDa, (g)divideDb  };
Binary MAX      = { (g)max,      (g)maxDa,    (g)maxDb     };
Binary MIN      = { (g)min,      (g)minDa,    (g)minDb     };

typedef struct optimizer {
    Real *m;           // SGD momentum or Adam first moment
    Real *v;           // Adam second moment
    int t;             // Timestep
    Real lr;           // learning rate
    Real decay;        // SGD momentum, or Adam beta1
    Real beta2;        // Adam second moment decay
    Real epsilon;      // Adam epsilon
    Real weightDecay;  // Adam weight decay
    Real biasM, biasV; // Adam bias corrections
    int inner;         // Entries one kernel covers
} Optimizer;

struct tensor {
    int  dimensions;
    int  *shape;
    int  *strides;
    void *entries;
    DataType type;        // Storage format of entries
    Real *gradients;      // Gradients for each entry, NULL if no autograd
    Optimizer *optimizer; // Optimizer state, NULL if no autograd
    Scale scale;          // Maps stored entries to Real, NULL if unscaled
    Tensor owner;         // Owner tensor if this is a view, NULL if not a view
    Operation *op;        // Operation that produced this tensor, NULL if none
    int references;       // Number of references
};

struct tensorArray {
    DynamicArray tensors;
};

// Axis functions
static void axesNew(Tensor t, int dimensions, const int *shape, const int *strides); // Allocate and set shape and strides
static void axesFree(Tensor t);                                                      // Free shape and strides

// Scale functions
static void scaleSlice(Scale *s, int group, int groups);                  // Keep the groups starting at group
static void scaleIndex(Scale *s, int group);                              // Collapse onto the single group
static void scaleFit(Scale *s, int group, Real low, Real high, Real max); // Map [low, high] onto a type reaching max
static void scaleOnes(Tensor A, Scale *s);                                // A unit scale, one group per row

// Key used for stochastic rounding at each step
static uint32_t stepKey(Tensor A) { return (uint32_t)A->optimizer->t * 0x9e3779b9u ^ (uint32_t)(uintptr_t)A->entries * 0xc2b2ae35u; }

// Pointer to the entry at given index, which must be byte-aligned for INT4 and INT2
static void *entry(Tensor A, size_t index) { return (unsigned char *)A->entries + DataSize(A->type, index); }

// Tensor accessor functions
int TensorDimensions(Tensor A)    { return A->dimensions;   }
int *TensorShape(Tensor A)        { return A->shape;        }
int *TensorStride(Tensor A)       { return A->strides;      }
void *TensorEntries(Tensor A)     { return A->entries;      }
DataType TensorDataType(Tensor A) { return A->type;         }
Real *TensorGradients(Tensor A)   { return A->gradients;    }
int TensorReference(Tensor A)     { return A->references;   }
Tensor TensorOwner(Tensor A)      { return A->owner;        }
Scale ScaleGet(Tensor A)                   { return A->scale; }
Operation *OperationGet(Tensor A)          { return A->op;    }
void OperationSet(Tensor A, Operation *op) { A->op = op;      }


static void axesNew(Tensor t, int dimensions, const int *shape, const int *strides)
{
    int axes = dimensions > 0 ? dimensions : 1;
    int *block = malloc(2 * (size_t)axes * sizeof(int));
    if (!block) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
    t->dimensions = dimensions;
    t->shape      = block;
    t->strides    = block + axes;
    for (int axis = 0; axis < dimensions; axis++) {
        t->shape[axis]   = shape[axis];
        t->strides[axis] = strides ? strides[axis] : 0;
    }
}

static void axesFree(Tensor t)
{
    free(t->shape);
    t->shape   = NULL;
    t->strides = NULL;
}


static void scaleSlice(Scale *s, int group, int groups)
{
    if (!s->values) return;
    if (s->stride) {
        s->values += group;
        if (s->zeros) s->zeros += group;
    }
    s->groups = groups;
}

static void scaleIndex(Scale *s, int group)
{
    if (!s->values) return;
    if (s->stride) {
        s->values += group;
        if (s->zeros) s->zeros += group;
    }
    s->groups = 1;
    s->stride = 0;
}

static void scaleOnes(Tensor A, Scale *s)
{
    int groups = A->dimensions >= 2 ? A->shape[0] : 1;
    ScaleNew(s, groups, A->dimensions >= 2 ? A->strides[0] : 0, DataMax(A->type) && !DataMin(A->type));
    for (int i = 0; i < groups; i++) s->values[i] = 1.0;
}

// Maps a group's entries, spanning [low, high], onto a type whose largest value is max.
static void scaleFit(Scale *s, int group, Real low, Real high, Real max)
{
    if (!s->zeros) {
        Real absmax = high > -low ? high : -low;
        s->values[group] = absmax > 0.0 ? absmax / max : 1.0;
        return;
    }
    if (low > 0.0)  low  = 0.0;
    if (high < 0.0) high = 0.0;
    Real value = high > low ? (high - low) / max : 1.0;
    Real zero  = (Real)(long)(-low / value + (Real)0.5); // -low / value is never negative
    s->values[group] = value;
    s->zeros[group]  = zero > max ? max : zero;
}

Tensor TensorView(Tensor A, int dimensions, const int *shape, const int *strides)
{
    Tensor view = malloc(sizeof(*view));
    if (!view) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
    *view = *A;
    axesNew(view, dimensions, shape, strides);
    view->owner = A;
    view->references = 1;
    A->references++;
    return view;
}

Real TensorGet(Tensor A, int index)
{
    return ScaleAt(A->scale, index) * (DataGet(A->entries, (size_t)index, A->type) - ScaleZeroAt(A->scale, index));
}

void TensorSet(Tensor A, int index, Real value)
{
    DataSet(A->entries, (size_t)index, A->type, value / ScaleAt(A->scale, index) + ScaleZeroAt(A->scale, index));
}

int TensorSize(Tensor A)
{
    int size = 1;
    for (int i = 0; i < A->dimensions; i++)
        size *= A->shape[i];
    return size;
}

int TensorArraySize(TensorArray ta)
{
    int total = 0;
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        total += TensorSize(*(Tensor *)DynamicArrayGet(ta->tensors, i));
    return total;
}

Tensor TensorNew(int dimensions, int *shape, DataType d, int autograd)
{
    if ((unsigned)d > FLOAT64) {
        fprintf(stderr, "TensorNew requires a valid DataType!\n");
        exit(EXIT_FAILURE);
    }
    Tensor t = calloc(1, sizeof(*t));
    if (!t) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
    t->type = d;
    t->references = 1;
    axesNew(t, dimensions, shape, NULL);
    if (dimensions > 0) {
        t->strides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; i--)
            t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }
    int size = TensorSize(t);
    size_t footprint = DataSize(d, (size_t)size);
    MemoryPool pool = MemoryPoolGet();
    t->entries   = MemoryPoolAlloc(pool, footprint);
    t->gradients = autograd ? MemoryPoolAlloc(pool, (size_t)size * sizeof(Real)) : NULL;
    t->optimizer = autograd ? calloc(1, sizeof(Optimizer)) : NULL;
    if (!t->entries || (autograd && (!t->gradients || !t->optimizer))) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    memset(t->entries, 0, footprint);
    return t;
}

TensorArray TensorArrayNew(void)
{
    TensorArray ta = malloc(sizeof(struct tensorArray));
    ta->tensors = DynamicArrayNew(sizeof(Tensor));
    return ta;
}

Tensor TensorArrayAdd(TensorArray ta, int dimensions, int *shape, DataType d, int autograd)
{
    Tensor t = TensorNew(dimensions, shape, d, autograd);
    if (autograd) memset(t->gradients, 0, (size_t)TensorSize(t) * sizeof(Real));
    DynamicArrayPush(ta->tensors, &t);
    return t;
}

void TensorFree(Tensor A)
{
    if (!A || --A->references > 0) return;
    Tensor owner = A->owner;
    if (!owner) {
        size_t footprint = 1;
        for (int i = 0; i < A->dimensions; i++)
            if (A->strides[i] > 0) footprint += (size_t)(A->shape[i] - 1) * A->strides[i];
        MemoryPool pool = MemoryPoolGet();
        MemoryPoolFree(pool, A->entries, DataSize(A->type, footprint));
        MemoryPoolFree(pool, A->gradients, footprint * sizeof(Real));
        ScaleFree(&A->scale);
        if (A->optimizer) {
            free(A->optimizer->m);
            free(A->optimizer->v);
            free(A->optimizer);
        }
        if (A->op) {
            TensorFree(A->op->A);
            TensorFree(A->op->B);
            free(A->op->context);
            free(A->op);
        }
    }
    axesFree(A);
    free(A);
    TensorFree(owner);
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
    int shape[A->dimensions > 0 ? A->dimensions : 1];
    for (int axis = 0; axis < A->dimensions; axis++) shape[axis] = A->shape[axis];
    shape[0] = length;
    Tensor t = TensorView(A, A->dimensions, shape, A->strides);
    int offset   = index * A->strides[0];
    t->entries   = entry(A, (size_t)offset);
    t->gradients = A->gradients ? A->gradients + offset : NULL;
    if (A->dimensions >= 2) scaleSlice(&t->scale, index, length);
    return t;
}

Tensor TensorIndex(Tensor A, int index)
{
    int dimensions = A->dimensions - 1;
    Tensor t = TensorView(A, dimensions, A->shape + 1, A->strides + 1);
    int offset   = index * A->strides[0];
    t->entries   = entry(A, (size_t)offset);
    t->gradients = A->gradients ? A->gradients + offset : NULL;
    if (A->dimensions >= 2) scaleIndex(&t->scale, index);
    return t;
}

Tensor TensorReshape(Tensor A, int dimensions, int *shape)
{
    if (!isContiguous(A->shape, A->strides, A->dimensions)) {
        fprintf(stderr, "TensorReshape requires a contiguous input!\n");
        exit(EXIT_FAILURE);
    }
    if (A->scale.values) {
        fprintf(stderr, "TensorReshape requires an unscaled input!\n");
        exit(EXIT_FAILURE);
    }
    int strides[dimensions > 0 ? dimensions : 1];
    if (dimensions > 0) {
        strides[dimensions - 1] = 1;
        for (int i = dimensions - 2; i >= 0; i--)
            strides[i] = strides[i + 1] * shape[i + 1];
    }
    return TensorView(A, dimensions, shape, strides);
}

Tensor TensorDetach(Tensor A)
{
    Tensor t = TensorView(A, A->dimensions, A->shape, A->strides);
    t->gradients = NULL;
    t->optimizer = NULL;
    t->op = NULL;
    return t;
}

Tensor TensorTranspose(Tensor A)
{
    int dims = A->dimensions;
    int shape[dims], strides[dims];
    for (int axis = 0; axis < dims; axis++) {
        shape[axis] = A->shape[axis];
        strides[axis] = A->strides[axis];
    }
    shape[dims - 2]   = A->shape[dims - 1];
    shape[dims - 1]   = A->shape[dims - 2];
    strides[dims - 2] = A->strides[dims - 1];
    strides[dims - 1] = A->strides[dims - 2];
    return TensorView(A, dims, shape, strides);
}

Tensor TensorRestride(Tensor A, int dimensions, int *shape, int *strides, int offset)
{
    if (A->scale.values) {
        fprintf(stderr, "TensorRestride requires an unscaled input!\n");
        exit(EXIT_FAILURE);
    }
    Tensor t = TensorView(A, dimensions, shape, strides);
    t->entries   = entry(A, (size_t)offset);
    t->gradients = A->gradients ? A->gradients + offset : NULL;
    return t;
}

Tensor TensorReduce(Binary fn, Real initial, Tensor A, int axis)
{
    Tensor indices = TensorNew(1, (int[]){A->shape[axis]}, FLOAT32, FALSE);
    Tensor out     = TensorScatter(fn, A, indices, initial, axis, 1);
    TensorFree(indices);
    return out;
}

Tensor TensorMap(Unary fn, Tensor A)                                      { return OperationMap(fn, A);                    }
Tensor TensorZip(Binary fn, Tensor A, Tensor B)                           { return OperationZip(fn, A, B);                 }
Tensor TensorMultiply(Tensor A, Tensor B)                                 { return OperationMultiply(A, B);                }
Tensor TensorAffine(Unary fn, Tensor A, Tensor bias, int inner)           { return OperationAffine(fn, A, bias, inner);    }
Tensor TensorGather(Tensor A, Tensor indices, int axis)                   { return OperationGather(A, indices, axis);      }
Tensor TensorScatter(Binary fn, Tensor A, Tensor i, Real e, int a, int n) { return OperationScatter(fn, A, i, e, a, n);    }
Tensor TensorConvolution(Tensor in, Tensor filter, int s, int p)          { return OperationConvolution(in, filter, s, p); }
Tensor TensorPool(Binary fn, Real e, Tensor in, int size, int s, int p)   { return OperationPool(fn, e, in, size, s, p);   }
Tensor TensorSoftmax(Tensor in, Real temperature)                         { return OperationSoftmax(in, temperature);      }
Tensor TensorCrossEntropyLoss(Tensor y, Tensor logits, Real temp)         { return OperationCrossEntropy(y, logits, temp); }
Tensor TensorMSE(Tensor y, Tensor y_hat)                                  { return OperationMSE(y, y_hat);                 }
void TensorBackward(Tensor loss)                                          { OperationBackward(loss);                       }

Tensor TensorAdd(Tensor A, Tensor B)                                      { return TensorZip(ADD, A, B);                   }
Tensor TensorSubtract(Tensor A, Tensor B)                                 { return TensorZip(SUBTRACT, A, B);              }
Tensor TensorHadamard(Tensor A, Tensor B)                                 { return TensorZip(MULTIPLY, A, B);              }
Tensor TensorEmbedding(Tensor table, Tensor indices)                      { return TensorGather(table, indices, 0);        }
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
    int inner    = conv->shape[2] * conv->shape[3];
    Tensor out   = TensorAffine(fn, conv, bias, inner);
    TensorFree(conv);
    return out;
}

Tensor TensorCopy(Tensor A)
{
    int contiguous = isContiguous(A->shape, A->strides, A->dimensions);
    if (A->gradients || (A->scale.values && !contiguous)) return TensorMap(IDENTITY, A);

    Tensor out = TensorNew(A->dimensions, A->shape, A->type, FALSE);
    int size   = TensorSize(A);

    if (A->scale.values) { // Contiguous, so the groups carry over as they are
        ScaleNew(&out->scale, A->scale.groups, A->scale.stride, A->scale.zeros != NULL);
        for (int c = 0; c < A->scale.groups; c++) {
            out->scale.values[c] = A->scale.values[c];
            if (out->scale.zeros) out->scale.zeros[c] = A->scale.zeros[c];
        }
        memcpy(out->entries, A->entries, DataSize(A->type, (size_t)size));
        return out;
    }
    if (contiguous) {
        memcpy(out->entries, A->entries, DataSize(A->type, (size_t)size));
        return out;
    }
    for (int i = 0; i < size; i++)
        DataConvert(out->entries, (size_t)i, A->type, A->entries,
                    (size_t)unravel(i, A->shape, A->strides, A->dimensions), A->type, 1);
    return out;
}

Tensor TensorScalarMultiply(Tensor A, Real scalar)
{
    if (!A->gradients && A->type != FLOAT32 && A->owner) {
        Tensor copy = TensorCopy(A);
        TensorFree(A);
        A = copy;
    }
    if (!A->gradients && A->type != FLOAT32) { // Scaling a quantized tensor only moves its scale
        int groups = A->dimensions >= 2 ? A->shape[0] : 1;
        if (!A->scale.values) {
            ScaleNew(&A->scale, groups, A->strides[0], FALSE);
            for (int c = 0; c < groups; c++) A->scale.values[c] = scalar;
        } else {
            for (int c = 0; c < A->scale.groups; c++) A->scale.values[c] *= scalar;
        }
        return A;
    }
    if (!A->gradients) {
        Real *e = A->entries;
        int size = TensorSize(A);
        int contiguous = isContiguous(A->shape, A->strides, A->dimensions);
        for (int i = 0; i < size; i++) {
            int off = contiguous ? i : unravel(i, A->shape, A->strides, A->dimensions);
            e[off] *= scalar;
        }
        return A;
    }
    Tensor s = TensorNew(0, NULL, FLOAT32, FALSE);
    TensorFill(s, scalar);
    Tensor out = TensorZip(MULTIPLY, A, s);
    TensorFree(s);
    TensorFree(A);
    return out;
}

void TensorFill(Tensor A, Real value)
{
    int size = TensorSize(A);
    if (A->type != FLOAT32 || A->scale.values) { // Scaled entries go through the mapping
        for (int i = 0; i < size; i++) TensorSet(A, i, value);
        return;
    }
    Real *e = A->entries;
    for (int i = 0; i < size; i++)
        e[i] = value;
}

Tensor TensorBroadcast(Tensor A, int axis, int n)
{
    if (axis < 0 || axis >= A->dimensions) {
        fprintf(stderr, "TensorBroadcast axis out of range!\n");
        exit(EXIT_FAILURE);
    }
    if (A->shape[axis] != 1) {
        fprintf(stderr, "TensorBroadcast axis must have size 1!\n");
        exit(EXIT_FAILURE);
    }
    Tensor indices = TensorNew(1, (int[]){n}, FLOAT32, FALSE);
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
    Tensor mask = TensorNew(2, (int[]){n, n}, FLOAT32, FALSE);
    Real *e = mask->entries;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            e[i * n + j] = (j <= i) ? 0.0f : -INFINITE;
    return mask;
}

Tensor TensorLogSoftmax(Tensor in, Real temperature)
{
    // Numerical stability trick
    Tensor inView       = TensorDetach(in);
    Tensor temp1        = TensorReduce(MAX, -INFINITE, inView, in->dimensions - 1);
    Tensor stableLogits = TensorZip(SUBTRACT, in, temp1);
    TensorFree(temp1);

    // Temperature scaling
    if (temperature != 1.0f)
        stableLogits = TensorScalarMultiply(stableLogits, 1.0f / temperature);

    // out = stableLogits - log(sum(exp(stableLogits)))
    Tensor A = TensorMap(EXP, stableLogits);
    Tensor B = TensorReduce(ADD, 0.0f, A, in->dimensions - 1);
    Tensor C = TensorMap(LOG, B);
    Tensor result = TensorZip(SUBTRACT, stableLogits, C);

    TensorFree(inView);
    TensorFree(stableLogits);
    TensorFree(A); TensorFree(B); TensorFree(C);
    return result;
}

Tensor TensorNLL(Tensor y, Tensor logP)
{
    int N = TensorSize(y) / y->shape[y->dimensions - 1];

    Tensor A   = TensorZip(MULTIPLY, y, logP);
    Tensor B   = TensorReduce(ADD, 0.0f, A, y->dimensions - 1);
    Tensor sum = TensorMap(NEGATE, B);

    while (TensorSize(sum) > 1) {
        int axis = 0;
        while (sum->shape[axis] == 1) axis++;
        Tensor next = TensorReduce(ADD, 0.0f, sum, axis);
        TensorFree(sum);
        sum = next;
    }
    Tensor scalar = TensorReshape(sum, 0, NULL);
    TensorFree(sum);
    sum = scalar;

    Tensor n = TensorNew(0, NULL, FLOAT32, FALSE);
    TensorFill(n, (Real)N);
    Tensor result = TensorZip(DIVIDE, sum, n);

    TensorFree(A); TensorFree(B); TensorFree(sum); TensorFree(n);
    return result;
}

Tensor TensorSample(Tensor logits, Real temperature)
{
    if (logits->type != FLOAT32 || logits->scale.values) {
        fprintf(stderr, "TensorSample requires unscaled FLOAT32 entries!\n");
        exit(EXIT_FAILURE);
    }
    Real *le       = logits->entries;
    int numClasses = logits->shape[logits->dimensions - 1];
    int n          = TensorSize(logits) / numClasses;

    int outDims = logits->dimensions - 1;
    int outShape[outDims > 0 ? outDims : 1];
    for (int i = 0; i < outDims; i++) outShape[i] = logits->shape[i];
    Tensor out = TensorNew(outDims, outShape, FLOAT32, 0);
    Real *oe   = out->entries;

    if (temperature == 0.0f) {
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int j = 1; j < numClasses; j++)
                if (le[i * numClasses + j] > le[i * numClasses + best]) best = j;
            oe[i] = (Real)best;
        }
    } else {
        Tensor P = TensorSoftmax(logits, temperature);
        Real *pe = P->entries;
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
    int aC = isContiguous(A->shape, A->strides, A->dimensions);
    int bC = isContiguous(B->shape, B->strides, B->dimensions);
    for (int i = 0; i < n; i++) {
        int aOff = aC ? i : unravel(i, A->shape, A->strides, A->dimensions);
        int bOff = bC ? i : unravel(i, B->shape, B->strides, B->dimensions);
        if (TensorGet(A, aOff) != TensorGet(B, bOff)) count++;
    }
    return count;
}

Tensor TensorShuffle(Tensor A, unsigned int seed)
{
    int n = A->shape[0];
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

    Tensor out = TensorNew(A->dimensions, A->shape, A->type, 0);
    if (A->scale.values) {
        int groups = A->dimensions >= 2 ? n : 1;
        ScaleNew(&out->scale, groups, out->strides[0], A->scale.zeros != NULL);
        for (int i = 0; i < groups; i++) {
            int from = A->dimensions >= 2 ? perm[i] : i;
            out->scale.values[i] = A->scale.values[from];
            if (out->scale.zeros) out->scale.zeros[i] = A->scale.zeros[from];
        }
    }
    int rowSize = TensorSize(A) / n;
    if (isContiguous(A->shape, A->strides, A->dimensions)) {
        // Each row is contiguous, so permute rows with a single copy each
        for (int i = 0; i < n; i++)
            DataConvert(out->entries, (size_t)i * rowSize, A->type,
                        A->entries, (size_t)perm[i] * rowSize, A->type, rowSize);
    } else {
        int innerDims = A->dimensions - 1;
        for (int i = 0; i < n; i++) {
            int srcRowOff = perm[i] * A->strides[0];
            for (int j = 0; j < rowSize; j++) {
                int innerOff = unravel(j, A->shape + 1, A->strides + 1, innerDims);
                Real v = DataGet(A->entries, (size_t)(srcRowOff + innerOff), A->type);
                DataSet(out->entries, (size_t)i * rowSize + j, out->type, v);
            }
        }
    }
    free(perm);
    return out;
}

void TensorXavier(Tensor A)
{
    int receptive = 1;
    for (int i = 2; i < A->dimensions; i++) receptive *= A->shape[i];
    int fanSum = (A->shape[0] + (A->dimensions > 1 ? A->shape[1] : 0)) * receptive;
    Real bound = VectorFirst(sqrte(VectorBcast((Real)6.0 / fanSum)));
    int size = TensorSize(A);

    if (DataMax(A->type)) {
        if (!A->scale.values) scaleOnes(A, &A->scale);
        for (int i = 0; i < A->scale.groups; i++)
            scaleFit(&A->scale, i, -bound, bound, (Real)DataMax(A->type));
    }
    if (A->type != FLOAT32 || A->scale.values) { // Scaled entries go through the mapping
        for (int i = 0; i < size; i++) TensorSet(A, i, randomNumber(-bound, bound));
        return;
    }
    Real *e = A->entries;
    for (int i = 0; i < size; i++)
        e[i] = randomNumber(-bound, bound);
}

Tensor TensorOneHot(Tensor labels, int numClasses)
{
    int outShape[labels->dimensions + 1];
    for (int i = 0; i < labels->dimensions; i++) outShape[i] = labels->shape[i];
    outShape[labels->dimensions] = numClasses;

    Tensor out = TensorNew(labels->dimensions + 1, outShape, FLOAT32, 0);
    Real *oe = out->entries;
    int numLabels = TensorSize(labels);

    for (int i = 0; i < numLabels; i++) {
        int off = unravel(i, labels->shape, labels->strides, labels->dimensions);
        oe[i * numClasses + (int)TensorGet(labels, off)] = 1.0f;
    }

    return out;
}

void TensorPrint(Tensor A)
{
    int size = TensorSize(A);
    int indices[A->dimensions > 0 ? A->dimensions : 1];

    if (size == 1) {
        printf("%g\n", TensorGet(A, 0));
        return;
    }

    int prevClose = 0;
    for (int i = 0; i < size; i++) {
        int temp = i;
        for (int d = A->dimensions - 1; d >= 0; d--) {
            indices[d] = temp % A->shape[d];
            temp /= A->shape[d];
        }

        int open = 0;
        for (int d = A->dimensions - 1; d >= 0; d--) {
            if (indices[d] == 0) open++; else break;
        }

        int close = 0;
        for (int d = A->dimensions - 1; d >= 0; d--) {
            if (indices[d] == A->shape[d] - 1) close++; else break;
        }

        if (i > 0) {
            for (int b = 0; b < prevClose; b++) printf("]");
            printf(",");
        }
        for (int b = 0; b < open; b++) printf("[");
        printf("%g", TensorGet(A, unravel(i, A->shape, A->strides, A->dimensions)));
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
    size_t data = DataSize(d, (size_t)TensorSize(t));
    if (fread(t->entries, 1, data, f) != data) {
        fprintf(stderr, "Cannot read file: %s\n", path);
        TensorFree(t);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    fclose(f);
    return t;
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
        Tensor t = *(Tensor *)DynamicArrayGet(ta->tensors, i);
        if (!isContiguous(t->shape, t->strides, t->dimensions)) {
            fprintf(stderr, "Cannot write non-contiguous tensor: %s\n", path);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        // Write data type
        int32_t type = (int32_t)t->type;
        if (fwrite(&type, sizeof(type), 1, f) != 1) {
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

        // Write tensor scale
        uint32_t groups = t->scale.values ? (uint32_t)t->scale.groups : 0;
        uint32_t zeros  = t->scale.zeros  ? groups : 0;
        if (fwrite(&groups, sizeof(groups), 1, f) != 1 ||
            fwrite(&zeros,  sizeof(zeros),  1, f) != 1) {
            fprintf(stderr, "Cannot write file: %s\n", path);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        for (uint32_t c = 0; c < groups; c++) {
            double perChannel = t->scale.values[c];
            if (fwrite(&perChannel, sizeof(perChannel), 1, f) != 1) {
                fprintf(stderr, "Cannot write file: %s\n", path);
                fclose(f);
                exit(EXIT_FAILURE);
            }
        }
        for (uint32_t c = 0; c < zeros; c++) {
            double zero = t->scale.zeros[c];
            if (fwrite(&zero, sizeof(zero), 1, f) != 1) {
                fprintf(stderr, "Cannot write file: %s\n", path);
                fclose(f);
                exit(EXIT_FAILURE);
            }
        }

        // Write tensor entries
        size_t data = DataSize(t->type, (size_t)TensorSize(t));
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
        // Read data type
        int32_t type;
        if (fread(&type, sizeof(type), 1, f) != 1 || (uint32_t)type > FLOAT64)
            goto error;

        // Read tensor shape
        int32_t dimensions;
        if (fread(&dimensions, sizeof(dimensions), 1, f) != 1 || dimensions < 0)
            goto error;

        int *shape = malloc((size_t)(dimensions > 0 ? dimensions : 1) * sizeof(int));
        if (!shape) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }

        for (int d = 0; d < dimensions; d++) {
            int32_t dim;
            if (fread(&dim, sizeof(dim), 1, f) != 1 || dim <= 0) {
                free(shape);
                goto error;
            }
            shape[d] = dim;
        }

        // Read scale count
        uint32_t groups, zeros;
        if (fread(&groups, sizeof(groups), 1, f) != 1 ||
            fread(&zeros,  sizeof(zeros),  1, f) != 1 ||
            (groups != 0 && groups != (uint32_t)(dimensions >= 2 ? shape[0] : 1)) ||
            (zeros  != 0 && zeros  != groups)) {
            free(shape);
            goto error;
        }

        // Check if tensor already exists, otherwise create a new one
        Tensor t;
        if (existing) {
            t = *(Tensor *)DynamicArrayGet(ta->tensors, i);
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
            t = created;
        }
        free(shape);
        ScaleFree(&t->scale);
        if (groups) {
            ScaleNew(&t->scale, (int)groups, t->strides[0], zeros != 0);
            for (uint32_t c = 0; c < groups; c++) {
                double perChannel;
                if (fread(&perChannel, sizeof(perChannel), 1, f) != 1)
                    goto error;
                t->scale.values[c] = (Real)perChannel;
            }
            for (uint32_t c = 0; c < zeros; c++) {
                double zero;
                if (fread(&zero, sizeof(zero), 1, f) != 1)
                    goto error;
                t->scale.zeros[c] = (Real)zero;
            }
        }

        // Read entries into the tensor
        size_t data = DataSize(t->type, (size_t)TensorSize(t));
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

/*
 * Algorithm 2 of Courbariaux et al. Please see: https://arxiv.org/pdf/1412.7024
 */
static void scaleUpdate(Tensor A, int base, int inner, int group, int overflow, int wouldOverflow)
{
    float rMax = 1.0 / 16.0;
    int widen   = (Real)overflow > rMax * inner;
    int tighten = !widen && (Real)wouldOverflow <= rMax * inner;
    if (!widen && !tighten) return;

    Real zero = ScaleZero(A->scale, group), wb[VECTOR_DIMS * VECTOR_CARD];
    DataType d = A->type;
    int chunk = VECTOR_DIMS * VECTOR_CARD;
    for (int c = 0; c < inner; c += chunk) {
        int nc = inner - c < chunk ? inner - c : chunk;
        DataConvert(wb, 0, FLOAT32, A->entries, (size_t)(base + c), d, nc);
        for (int k = 0; k < nc; k++) wb[k] = zero + (widen ? (wb[k] - zero) / 2 : (wb[k] - zero) * 2);
        DataConvertStochastic(A->entries, (size_t)(base + c), d, wb, 0, FLOAT32, nc, stepKey(A) ^ 0x2545f491u);
    }
    A->scale.values[group] = widen ? A->scale.values[group] * 2 : A->scale.values[group] / 2;
}

static int blockInner(Tensor A, int size)
{
    int groups = A->scale.values ? A->scale.groups : 1;
    if (groups > 1 && A->scale.stride) {
        if (A->scale.stride % DataPerByte(A->type)) return size;
        return A->scale.stride;
    }
    if (DataMax(A->type)) return size;

    int workers = ThreadPoolSize(ThreadPoolGet());
    if (workers < 1) workers = 1;

    int inner = (size + workers - 1) / workers;
    inner = (inner + VECTOR_DIMS - 1) / VECTOR_DIMS * VECTOR_DIMS;
    return inner < size ? inner : size;
}

/*
 * ECO: each step's rounding residual is kept in the first moment, and the weights store the
 * updated weight w' rounded stochastically. Please see: https://arxiv.org/abs/2601.22101
 */
static int ecoTrains(Tensor A, Real lr, Real decay)
{
    return DataMin(A->type) < 0 && !A->scale.zeros && lr > 0.0 && decay > 0.0 && decay < 1.0;
}

static void ecoKernel(Tensor A, int base, int n, int sgd)
{
    Optimizer *o = A->optimizer;
    int rowLen = A->scale.groups > 1 ? A->scale.stride : TensorSize(A), chunk = VECTOR_DIMS * VECTOR_CARD;
    Real bm = sgd ? 1.0 : o->biasM, C = o->lr * o->decay / (bm * (1 - o->decay));
    Vector b1 = VectorBcast(o->decay), c1 = VectorBcast(1 - o->decay);
    Vector b2 = VectorBcast(o->beta2), c2 = VectorBcast(1 - o->beta2);
    Vector biasV = VectorBcast(o->biasV), epsv = VectorBcast(o->epsilon), wdv = VectorBcast(o->weightDecay);
    Vector rate = VectorBcast(o->lr), bmv = VectorBcast(bm), Cv = VectorBcast(C);
    Vector low = VectorBcast((Real)DataMin(A->type)), high = VectorBcast((Real)DataMax(A->type));
    uint32_t key = stepKey(A);

    Real eb[VECTOR_DIMS * VECTOR_CARD];
    for (int row = base; row < base + n; row += rowLen) {
        int group = row / rowLen;
        Real scale = A->scale.values[group], next = scale;
        Vector stat = VectorZero(), nextv = VectorZero(), inverse = VectorZero();
        for (int pass = 0; pass < 2; pass++) {
            for (int c = 0; c < rowLen; c += chunk) {
                int nc = rowLen - c < chunk ? rowLen - c : chunk;
                size_t e0 = (size_t)row + (size_t)c;
                DataConvert(eb, 0, FLOAT32, A->entries, e0, A->type, nc);
                for (int j = 0; j < nc; j += VECTOR_DIMS) {
                    int cnt = nc - j < VECTOR_DIMS ? nc - j : VECTOR_DIMS;
                    Real *mp = o->m + e0 + j, *vp = sgd ? NULL : o->v + e0 + j;
                    Vector w = VectorMul(VectorBcast(scale), VectorLoadPartial(eb + j, cnt));
                    Vector m = VectorLoadPartial(mp, cnt), v = vp ? VectorLoadPartial(vp, cnt) : VectorZero();
                    if (pass == 0) {
                        Vector g = VectorLoadPartial(A->gradients + e0 + j, cnt);
                        m = sgd ? VectorFma(b1, m, g) : VectorFma(b1, m, VectorMul(c1, g));
                        VectorStorePartial(mp, m, cnt);
                        if (vp) {
                            v = VectorFma(b2, v, VectorMul(c2, VectorMul(g, g)));
                            VectorStorePartial(vp, v, cnt);
                        }
                    }

                    // SGD: w' = w - lr m with C = lr b1 / (1 - b1)
                    // Adam: w' = w - lr (m / (bm d) + wd w) with C = lr b1 / (bm (1 - b1))
                    Vector d = sgd ? VectorBcast(1.0) : VectorAdd(sqrte(VectorDiv(v, biasV)), epsv);
                    Vector wp = sgd ? VectorFnmadd(rate, m, w)
                                    : VectorFnmadd(rate, VectorAdd(VectorDiv(m, VectorMul(bmv, d)), VectorMul(wdv, w)), w);
                    if (pass == 0) {
                        Vector a = VectorMax(wp, VectorZero() - wp);
                        stat = A->type == INT2 ? VectorAdd(stat, a) : VectorMax(stat, a);
                        continue;
                    }

                    Vector dithered = VectorAdd(VectorMul(wp, inverse), VectorUniformAt(key, e0 + (size_t)j));
                    Vector code = VectorMin(VectorMax(VectorFloor(dithered), low), high);
                    VectorStorePartial(mp, VectorSub(m, VectorMul(d, VectorDiv(VectorFnmadd(nextv, code, wp), Cv))), cnt);
                    VectorStorePartial(eb + j, code, cnt);
                }
                if (pass == 1) DataConvert(A->entries, e0, A->type, eb, 0, FLOAT32, nc);
            }

            // The row's scale from its updated weights: INT2 is ternary and takes their absmean,
            // wider types map the absmax onto the type's largest value
            if (pass == 0) {
                if (A->type == INT2) {
                    next = VectorHsum(stat) / rowLen + (Real)1e-8;
                } else {
                    Real absmax = 0.0;
                    for (int l = 0; l < VECTOR_DIMS; l++) if (stat[l] > absmax) absmax = stat[l];
                    if (absmax > 0.0) next = absmax / (Real)DataMax(A->type);
                }
                nextv = VectorBcast(next);
                inverse = VectorBcast(1 / next);
            }
        }
        A->scale.values[group] = next;
    }
}

static void ecoSGDBlock(void *arg, int start, int end)
{
    Tensor A = arg;
    int size = TensorSize(A), inner = A->optimizer->inner;

    for (int block = start; block < end; block++) {
        int base = block * inner;
        ecoKernel(A, base, size - base < inner ? size - base : inner, 1);
    }
}

static void ecoAdamBlock(void *arg, int start, int end)
{
    Tensor A = arg;
    int size = TensorSize(A), inner = A->optimizer->inner;

    for (int block = start; block < end; block++) {
        int base = block * inner;
        ecoKernel(A, base, size - base < inner ? size - base : inner, 0);
    }
}

static inline void sgdKernel(Tensor A, DataType d, int base, int n, int group,
                          int *overflowOut, int *wouldOut)
{
    Optimizer *o = A->optimizer;
    Real *m = o->decay != 0.0 ? o->m : NULL;
    int64_t qMax = DataMax(d);
    int inPlace = d == FLOAT32 && sizeof(Real) == 4;
    Real scale = ScaleValue(A->scale, group), zero = ScaleZero(A->scale, group);
    Vector decay = VectorBcast(o->decay);
    Vector rate = VectorBcast(scale > 0.0 ? o->lr / scale : 0.0);
    Vector low = VectorBcast((Real)DataMin(d)), high = VectorBcast((Real)qMax);
    Vector zeroed = VectorBcast(zero);
    Vector upv = VectorBcast((Real)qMax - zero), downv = VectorBcast(zero - (Real)DataMin(d));
    Vector overflow = VectorZero(), wouldOverflow = VectorZero();
    uint32_t key = stepKey(A);

    Real eb[VECTOR_DIMS * VECTOR_CARD];
    int chunk = VECTOR_DIMS * VECTOR_CARD;
    for (int c = 0; c < n; c += chunk) {
        int nc = n - c < chunk ? n - c : chunk;
        Real *entries = inPlace ? (Real *)A->entries + base + c : eb;
        if (!inPlace) DataConvert(eb, 0, FLOAT32, A->entries, (size_t)(base + c), d, nc);
        for (int j = 0; j < nc; j += VECTOR_DIMS) {
            int cnt = nc - j < VECTOR_DIMS ? nc - j : VECTOR_DIMS;
            Real *mp = m ? m + base + c + j : NULL;

            Vector step = VectorLoadPartial(A->gradients + base + c + j, cnt);
            if (mp) { step = VectorFma(decay, VectorLoadPartial(mp, cnt), step); VectorStorePartial(mp, step, cnt); }
            Vector w = VectorFnmadd(rate, step, VectorLoadPartial(entries + j, cnt));
            if (qMax) {
                Vector in = VectorMin(VectorMax(w, low), high);
                Vector over = VectorSub(w, in);
                overflow = VectorAdd(overflow, VectorGt0(VectorMax(over, VectorZero() - over)));
                w = in;
                Vector twice = VectorAdd(VectorSub(w, zeroed), VectorSub(w, zeroed));
                wouldOverflow = VectorAdd(wouldOverflow,
                                          VectorGt0(VectorMax(VectorSub(twice, upv),
                                                              VectorSub(VectorZero() - downv, twice))));
            }
            VectorStorePartial(entries + j, w, cnt);
        }
        if (!inPlace) DataConvertStochastic(A->entries, (size_t)(base + c), d, eb, 0, FLOAT32, nc, key);
    }

    *overflowOut = (int)VectorHsum(overflow);
    *wouldOut = (int)VectorHsum(wouldOverflow);
}

static void sgdBlock(void *arg, int start, int end)
{
    Tensor A = arg;
    int size = TensorSize(A), inner = A->optimizer->inner;

    for (int block = start; block < end; block++) {
        int base = block * inner;
        int n = size - base < inner ? size - base : inner;
        int group = A->scale.stride ? base / A->scale.stride : 0;
        int overflow = 0, would = 0;

        switch (A->type) {
            case FLOAT32: sgdKernel(A, FLOAT32, base, n, group, &overflow, &would); break;
            case INT8:    sgdKernel(A, INT8,    base, n, group, &overflow, &would); break;
            case UINT8:   sgdKernel(A, UINT8,   base, n, group, &overflow, &would); break;
            case INT16:   sgdKernel(A, INT16,   base, n, group, &overflow, &would); break;
            case INT32:   sgdKernel(A, INT32,   base, n, group, &overflow, &would); break;
            case INT4:    sgdKernel(A, INT4,    base, n, group, &overflow, &would); break;
            case INT2:    sgdKernel(A, INT2,    base, n, group, &overflow, &would); break;
            default:      sgdKernel(A, A->type, base, n, group, &overflow, &would); break;
        }

        if (DataMax(A->type)) scaleUpdate(A, base, n, group, overflow, would);
    }
}


void TensorSGD(Tensor A, Real lr, Real momentum)
{
    Optimizer *optimizer = A->optimizer;
    if (!optimizer) return; // skip tensors without autograd
    int size = TensorSize(A);
    int64_t qMax = DataMax(A->type);
    if (qMax && !A->scale.values) scaleOnes(A, &A->scale);

    if (momentum != 0.0 && !optimizer->m) {
        optimizer->m = calloc((size_t)size, sizeof(Real));
        if (!optimizer->m) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    optimizer->t++;
    optimizer->lr = lr;
    optimizer->decay = momentum;
    int inner = optimizer->inner = blockInner(A, size);
    if (ecoTrains(A, lr, momentum)) OperationDispatch(ecoSGDBlock, A, (size + inner - 1) / inner, (long)size * 30);
    else OperationDispatch(sgdBlock, A, (size + inner - 1) / inner, (long)size * 10);
}

void TensorArraySGD(TensorArray ta, Real lr, Real momentum)
{
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        TensorSGD(*(Tensor *)DynamicArrayGet(ta->tensors, i), lr, momentum);
}

static inline void adamKernel(Tensor A, DataType d, int base, int n, int group,
                           int *overflowOut, int *wouldOut)
{
    Optimizer *o = A->optimizer;
    Real *m = o->m, *v = o->v, *g = A->gradients;
    int64_t qMax = DataMax(d);
    int inPlace = d == FLOAT32 && sizeof(Real) == 4;
    Real scale = ScaleValue(A->scale, group), zero = ScaleZero(A->scale, group);
    Vector b1 = VectorBcast(o->decay), c1 = VectorBcast(1.0 - o->decay);
    Vector b2 = VectorBcast(o->beta2), c2 = VectorBcast(1.0 - o->beta2);
    Vector biasM = VectorBcast(o->biasM), biasV = VectorBcast(o->biasV);
    Vector epsv = VectorBcast(o->epsilon), wdv = VectorBcast(o->weightDecay);
    Vector rate = VectorBcast(scale > 0.0 ? o->lr / scale : 0.0);
    Vector scaled = VectorBcast(scale), zeroed = VectorBcast(zero);
    Vector low = VectorBcast((Real)DataMin(d)), high = VectorBcast((Real)qMax);
    Vector upv = VectorBcast((Real)qMax - zero), downv = VectorBcast(zero - (Real)DataMin(d));
    Vector overflow = VectorZero(), wouldOverflow = VectorZero();
    uint32_t key = stepKey(A);

    Real eb[VECTOR_DIMS * VECTOR_CARD];
    int chunk = VECTOR_DIMS * VECTOR_CARD;
    for (int c = 0; c < n; c += chunk) {
        int nc = n - c < chunk ? n - c : chunk;
        Real *entries = inPlace ? (Real *)A->entries + base + c : eb;
        if (!inPlace) DataConvert(eb, 0, FLOAT32, A->entries, (size_t)(base + c), d, nc);
        for (int j = 0; j < nc; j += VECTOR_DIMS) {
            int cnt = nc - j < VECTOR_DIMS ? nc - j : VECTOR_DIMS;
            Real *gp = g + base + c + j, *mp = m + base + c + j, *vp = v + base + c + j;

            Vector gv = VectorLoadPartial(gp, cnt);
            Vector mv = VectorFma(b1, VectorLoadPartial(mp, cnt), VectorMul(c1, gv));
            Vector vv = VectorFma(b2, VectorLoadPartial(vp, cnt), VectorMul(c2, VectorMul(gv, gv)));
            VectorStorePartial(mp, mv, cnt);
            VectorStorePartial(vp, vv, cnt);

            Vector e    = VectorLoadPartial(entries + j, cnt);
            Vector mHat = VectorDiv(mv, biasM);
            Vector vHat = VectorDiv(vv, biasV);
            Vector step = VectorAdd(VectorDiv(mHat, VectorAdd(sqrte(vHat), epsv)),
                                    VectorMul(wdv, VectorMul(scaled, VectorSub(e, zeroed))));
            Vector w = VectorFnmadd(rate, step, e);
            if (qMax) {
                Vector in = VectorMin(VectorMax(w, low), high);
                Vector over = VectorSub(w, in);
                overflow = VectorAdd(overflow, VectorGt0(VectorMax(over, VectorZero() - over)));
                w = in;
                Vector twice = VectorAdd(VectorSub(w, zeroed), VectorSub(w, zeroed));
                wouldOverflow = VectorAdd(wouldOverflow,
                                          VectorGt0(VectorMax(VectorSub(twice, upv),
                                                              VectorSub(VectorZero() - downv, twice))));
            }
            VectorStorePartial(entries + j, w, cnt);
        }
        if (!inPlace) DataConvertStochastic(A->entries, (size_t)(base + c), d, eb, 0, FLOAT32, nc, key);
    }

    *overflowOut = (int)VectorHsum(overflow);
    *wouldOut = (int)VectorHsum(wouldOverflow);
}

static void adamBlock(void *arg, int start, int end)
{
    Tensor A = arg;
    int size = TensorSize(A), inner = A->optimizer->inner;

    for (int block = start; block < end; block++) {
        int base = block * inner;
        int n = size - base < inner ? size - base : inner;
        int group = A->scale.stride ? base / A->scale.stride : 0;
        int overflow = 0, would = 0;

        switch (A->type) {
            case FLOAT32: adamKernel(A, FLOAT32, base, n, group, &overflow, &would); break;
            case INT8:    adamKernel(A, INT8,    base, n, group, &overflow, &would); break;
            case UINT8:   adamKernel(A, UINT8,   base, n, group, &overflow, &would); break;
            case INT16:   adamKernel(A, INT16,   base, n, group, &overflow, &would); break;
            case INT32:   adamKernel(A, INT32,   base, n, group, &overflow, &would); break;
            case INT4:    adamKernel(A, INT4,    base, n, group, &overflow, &would); break;
            case INT2:    adamKernel(A, INT2,    base, n, group, &overflow, &would); break;
            default:      adamKernel(A, A->type, base, n, group, &overflow, &would); break;
        }

        if (DataMax(A->type)) scaleUpdate(A, base, n, group, overflow, would);
    }
}

void TensorAdam(Tensor A, Real lr, Real beta1, Real beta2, Real epsilon, Real weightDecay)
{
    Optimizer *optimizer = A->optimizer;
    if (!optimizer) return; // skip tensors without autograd
    int size = TensorSize(A);
    int64_t qMax = DataMax(A->type);
    if (qMax && !A->scale.values) scaleOnes(A, &A->scale);

    if (!optimizer->m) {
        optimizer->m = calloc((size_t)size, sizeof(Real));
        optimizer->v = calloc((size_t)size, sizeof(Real));
        if (!optimizer->m || !optimizer->v) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    optimizer->t++;
    optimizer->lr = lr;
    optimizer->decay = beta1;
    optimizer->beta2 = beta2;
    optimizer->epsilon = epsilon;
    optimizer->weightDecay = weightDecay;
    optimizer->biasM = 1.0 - powi(beta1, optimizer->t);
    optimizer->biasV = 1.0 - powi(beta2, optimizer->t);

    int inner = optimizer->inner = blockInner(A, size);
    if (ecoTrains(A, lr, beta1)) OperationDispatch(ecoAdamBlock, A, (size + inner - 1) / inner, (long)size * 40);
    else OperationDispatch(adamBlock, A, (size + inner - 1) / inner, (long)size * 16);
}

void TensorArrayAdam(TensorArray ta, Real lr, Real beta1, Real beta2, Real epsilon, Real weightDecay)
{
    for (int i = 0; i < DynamicArraySize(ta->tensors); i++)
        TensorAdam(*(Tensor *)DynamicArrayGet(ta->tensors, i), lr, beta1, beta2, epsilon, weightDecay);
}

void TensorBPETrain(Tensor corpus, Tensor merges)
{
    int targetMerges = TensorShape(merges)[0];
    int n = TensorSize(corpus);
    int *ids = malloc((size_t)n * sizeof *ids);
    for (int i = 0; i < n; i++) ids[i] = (int)TensorGet(corpus, i);

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
        TensorSet(merges, 2 * numMerges, (Real)a);
        TensorSet(merges, 2 * numMerges + 1, (Real)b);

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
    int targetMerges = TensorShape(merges)[0];
    int n = TensorSize(corpus);
    int *ids = malloc((size_t)n * sizeof *ids);
    for (int i = 0; i < n; i++) ids[i] = (int)TensorGet(corpus, i);
    HashMap rankMap = HashMapNew(1024);
    for (int r = 0; r < targetMerges; r++)
        HashMapStore(rankMap, packi((int)TensorGet(merges, 2 * r), (int)TensorGet(merges, 2 * r + 1)), r);

    // Greedily merges the pair that the merges tensor (tokenizer) learned earliest, until none are left
    while (TRUE) {
        long best = -1;
        for (int i = 0; i < n - 1; i++) {
            long r = HashMapGet(rankMap, packi(ids[i], ids[i + 1]));
            if (r >= 0 && (best < 0 || r < best)) best = r;
        }
        if (best < 0) break;

        int a = (int)TensorGet(merges, 2 * (int)best);
        int b = (int)TensorGet(merges, 2 * (int)best + 1);
        int w = 0;
        for (int i = 0; i < n; ) {
            if (i < n - 1 && ids[i] == a && ids[i + 1] == b) { ids[w++] = 256 + (int)best; i += 2; }
            else ids[w++] = ids[i++];
        }
        n = w;
    }

    Tensor tokens = TensorNew(1, (int[]){n}, FLOAT32, FALSE);
    for (int i = 0; i < n; i++) TensorSet(tokens, i, (Real)ids[i]);
    free(ids);
    HashMapFree(rankMap);
    return tokens;
}

Tensor TensorBPEDecode(Tensor tokens, Tensor merges)
{
    int n = TensorSize(tokens);
    int vocabSize = 256 + TensorShape(merges)[0];

    // Decode each token by walking the merge tree with a stack
    Stack stack = StackNew(sizeof(int));
    DynamicArray out = DynamicArrayNew(sizeof(unsigned char));
    for (int i = 0; i < n; i++) {
        int id = (int)TensorGet(tokens, i);
        if (id < 0 || id >= vocabSize) continue;
        StackPush(stack, &id);
        int t;
        while (StackPop(stack, &t)) {
            if (t < 256) {
                unsigned char b = (unsigned char)t;
                DynamicArrayPush(out, &b);
            } else {
                int r = t - 256;
                int right = (int)TensorGet(merges, 2 * r + 1);
                int left  = (int)TensorGet(merges, 2 * r);
                StackPush(stack, &right);
                StackPush(stack, &left);
            }
        }
    }
    StackFree(stack);

    int total = DynamicArraySize(out);
    Tensor bytes = TensorNew(1, (int[]){total}, FLOAT32, FALSE);
    for (int i = 0; i < total; i++)
        TensorSet(bytes, i, (Real)*(unsigned char *)DynamicArrayGet(out, i));
    DynamicArrayFree(out);
    return bytes;
}
