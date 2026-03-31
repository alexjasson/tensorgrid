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

// Univariate/Bivariate functions to be applied to tensors
extern Univariate RELU;
extern Univariate SIGMOID;
extern Univariate LOG;
extern Univariate EXP;
extern Univariate NEGATE;
extern Univariate IDENTITY;
extern Bivariate  ADD;
extern Bivariate  SUBTRACT;
extern Bivariate  MULTIPLY;
extern Bivariate  DIVIDE;
extern Bivariate  MAX;


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

static inline Tensor TensorCopy(Tensor in)              { return TensorMap(IDENTITY, in); }
static inline Tensor TensorAdd(Tensor A, Tensor B)      { return TensorZip(ADD,      A, B); }
static inline Tensor TensorSubtract(Tensor A, Tensor B) { return TensorZip(SUBTRACT, A, B); }
static inline Tensor TensorHadamard(Tensor A, Tensor B) { return TensorZip(MULTIPLY, A, B); }


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
