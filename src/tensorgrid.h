#ifndef TENSORGRID_H
#define TENSORGRID_H

#include <math.h>

#define MAX_DIMENSIONS 8
#define TRUE 1
#define FALSE 0

typedef float Entry;                        // Change data type of entry as needed
typedef Entry (*UnivariateFn)(Entry);       // y = f(a) on an entry in A
typedef Entry (*BivariateFn)(Entry, Entry); // y = f(a,b) on entries in A and B
typedef struct { UnivariateFn f; UnivariateFn da; } Univariate;              // y = f(a) and dy/da
typedef struct { BivariateFn f; BivariateFn da; BivariateFn db; } Bivariate; // y = f(a,b), dy/da and dy/db

// Tensor representation
typedef struct {
    int dimensions;
    int shape[MAX_DIMENSIONS];
    int strides[MAX_DIMENSIONS];
    Entry *entries;
    Entry *gradients; // NULL if no autograd
    int   *refcount;  // Number of references, NULL for views
} Tensor;

// Univariate functions and derivates
static Entry relu(Entry a)     { return a > 0.0f ? a : 0.0f; }
static Entry drelu(Entry a)    { return a > 0.0f ? 1.0f : 0.0f; }
static Entry sigmoid(Entry a)  { return 1.0f / (1.0f + expf(-a)); }
static Entry dsigmoid(Entry a) { return sigmoid(a) * (1.0f - sigmoid(a)); }
static Entry loge(Entry a)     { return logf(a); }
static Entry dloge(Entry a)    { return 1.0f / a; }
static Entry negate(Entry a)   { return -a; }
static Entry identity(Entry a) { return a; }
static Entry one(Entry a)      { (void)a; return  1.0f; }
static Entry minusOne(Entry a) { (void)a; return -1.0f; }
static Entry zero(Entry a)     { (void)a; return 0.0f; }
static Entry expe(Entry a)     { return expf(a); }

// Bivariate functions and partial derivatives
static Entry add(Entry a, Entry b)       { return a + b; }
static Entry subtract(Entry a, Entry b)  { return a - b; }
static Entry multiply(Entry a, Entry b)  { return a * b; }
static Entry divide(Entry a, Entry b)    { return a / b; }
static Entry max(Entry a, Entry b)       { return a > b ? a : b; }
static Entry first(Entry a, Entry b)     { (void)b; return a; }
static Entry second(Entry a, Entry b)    { (void)a; return b; }
static Entry ones(Entry a, Entry b)      { (void)a; (void)b; return 1.0f; }
static Entry minusOnes(Entry a, Entry b) { (void)a; (void)b; return -1.0f; }
static Entry divideDa(Entry a, Entry b)  { (void)a; return 1.0f / b; }
static Entry divideDb(Entry a, Entry b)  { return -a / (b * b); }
static Entry maxDa(Entry a, Entry b)     { return a >= b ? 1.0f : 0.0f; }
static Entry maxDb(Entry a, Entry b)     { return b > a ? 1.0f : 0.0f; }

// Univariate/Bivariate functions and their derivatives/partial derivates - add more as needed
static Univariate RELU     = { relu,     drelu    };
static Univariate SIGMOID  = { sigmoid,  dsigmoid };
static Univariate LOG      = { loge,     dloge    };
static Univariate EXP      = { expe,     expe     };
static Univariate NEGATE   = { negate,   minusOne };
static Univariate IDENTITY = { identity, one      };
static Bivariate  ADD      = { add,      ones,      ones      };
static Bivariate  SUBTRACT = { subtract, ones,      minusOnes };
static Bivariate  MULTIPLY = { multiply, second,    first     };
static Bivariate  DIVIDE   = { divide,   divideDa,  divideDb  };
static Bivariate  MAX      = { max,      maxDa,     maxDb     };


/*========================================================================
 *                         Memory management
 *========================================================================*/

/*
 * Allocates a new tensor with all entries as 0.
 */
Tensor TensorNew(int dimensions, int *shape, int autograd);

/*
 * Frees a tensor from memory.
 */
void TensorFree(Tensor A);

/*
 * Reads a .npy file and returns it as a tensor. Data is cast to the Entry
 * data type (most likely float32).
 */
Tensor TensorRead(char *path);

/*
 * Writes a tensor to a .npy file.
 */
void TensorWrite(Tensor A, char *path);


/*========================================================================
 *                         Shape, views and access
 *========================================================================*/

/*
 * Returns the total number of entries in the tensor.
 */
int TensorSize(Tensor A);

/*
 * Returns a view with a new shape. Total size must be unchanged. Shares memory
 * with A, use with caution.
 */
Tensor TensorView(Tensor A, int dimensions, int *shape, int autograd);

/*
 * Returns a view of A[index] along the given axis (dimensions reduced by 1).
 */
Tensor TensorSlice(Tensor A, int axis, int index);

/*
 * Returns a view with the last two dimensions transposed.
 */
Tensor TensorTranspose(Tensor A);

/*
 * Returns the entry at the given N-dimensional index.
 */
Entry TensorGet(Tensor A, int *indices);

/*
 * Sets the entry at the given N-dimensional index.
 */
void TensorSet(Tensor A, int *indices, Entry value);


/*========================================================================
 *                              Operations
 *========================================================================*/

/*
 * out = f(A).
 */
Tensor TensorMap(Univariate f, Tensor A);

/*
 * out = f(A, B).
 */
Tensor TensorZip(Bivariate f, Tensor A, Tensor B);

/*
 * Reduces A along the given axis using f, starting from initial.
 * A:   [..., m, ...]
 * out: [..., 1, ...]
 */
Tensor TensorReduce(Bivariate f, Entry initial, Tensor A, int axis);

/*
 * Broadcasts the last dimension of A from 1 to n.
 * A:   [..., 1]
 * out: [..., n]
 */
Tensor TensorBroadcast(Tensor A, int n);

/*
 * Multiply A by B using A's last two dimensions as matrices.
 * A:   [..., m, k]
 * B:   [k, n]
 * out: [..., m, n]
 */
Tensor TensorMultiply(Tensor A, Tensor B);

static Tensor TensorCopy(Tensor in)              { return TensorMap(IDENTITY, in); }
static Tensor TensorAdd(Tensor A, Tensor B)      { return TensorZip(ADD,      A, B); }
static Tensor TensorSubtract(Tensor A, Tensor B) { return TensorZip(SUBTRACT, A, B); }
static Tensor TensorHadamard(Tensor A, Tensor B) { return TensorZip(MULTIPLY, A, B); }


/*========================================================================
 *                                Utilities
 *========================================================================*/

/*
 * Xavier initialize a tensor using the last two dimensions as fan_in/fan_out.
 */
void TensorXavier(Tensor A);

/*
 * Prints a tensor to stdout. Tensors of size 1 are printed as scalars.
 */
void TensorPrint(Tensor A);

/*
 * Softmax over the last dimension.
 * out = exp(in / T) ⊘ sum(exp(in / T))
 *
 * in:  Input logits    [..., n]
 * out: Probabilities   [..., n]
 */
Tensor TensorSoftmax(Tensor in, float temperature);

/*
 * Cross entropy loss over the last dimension.
 * out = -sum(y ⊙ log(y_hat))
 *
 * y_hat: Predicted probabilities [..., n]
 * y:     Target distribution     [..., n]
 * out:   Cross entropy loss      [..., 1]
 */
Tensor TensorCrossEntropyLoss(Tensor y, Tensor y_hat);

/*
 * Applies softmax to output logits and samples an index using inverse CDF sampling.
 */
int TensorSample(Tensor output, float temperature);

/*
 * Starts with loss tensor, takes mean for each entry, then propogates gradients
 * backwards through all recorded operations. All gradients are zero'd first and
 * then accumulated during the backward pass.
 */
void TensorBackward(Tensor loss);

/*
 * A -= lr * gradients. Must be called after TensorBackward.
 */
void TensorSGD(Tensor A, float lr);

#endif
