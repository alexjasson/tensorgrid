#ifndef TENSORGRID_H
#define TENSORGRID_H

#define MAX_DIMENSIONS 8
#define TRUE 1
#define FALSE 0

typedef float Entry;                        // Change data type of entry as needed
typedef Entry (*UnivariateFn)(Entry);       // y = f(a) on an entry in A
typedef Entry (*BivariateFn)(Entry, Entry); // y = f(a,b) on entries in A and B
typedef struct { UnivariateFn f; UnivariateFn da; } Univariate;              // y = f(a) and dy/da
typedef struct { BivariateFn f; BivariateFn da; BivariateFn db; } Bivariate; // y = f(a,b), dy/da and dy/db

typedef struct optimizer Optimizer;

// Tensor representation
typedef struct {
    int dimensions;
    int shape[MAX_DIMENSIONS];
    int strides[MAX_DIMENSIONS];
    Entry *entries;
    Entry *gradients;      // NULL if no autograd
    Optimizer *optimizer;  // Optimizer state, NULL if no autograd
    int   *refcount;       // Number of references, NULL for views
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
 * Note: for a 0-dimensional tensor (scalar), pass shape = NULL.
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
 * Returns the size of the tensor along the given axis.
 */
int TensorShape(Tensor A, int axis);

/*
 * Returns a view with a new shape. Total size must be unchanged. Shares memory
 * with A, use with caution.
 */
Tensor TensorView(Tensor A, int dimensions, int *shape, int autograd);

/*
 * Returns A with a new shape. Total size must be unchanged. Does not allocate memory.
 */
Tensor TensorReshape(Tensor A, int dimensions, int *shape);

/*
 * Returns a view of length entries starting at index along the given axis.
 */
Tensor TensorSlice(Tensor A, int axis, int index, int length);

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
 * The reduced axis is dropped from the output.
 * A:   [..., m, ...]
 * out: [..., ...]
 */
Tensor TensorReduce(Bivariate f, Entry initial, Tensor A, int axis);

/*
 * Inserts a new axis of size n at the given position, replicating A along it.
 * A:   [...]
 * out: [..., n, ...]
 */
Tensor TensorBroadcast(Tensor A, int axis, int n);

/*
 * Multiply A by B using the last two dimensions as matrices.
 * A:   [..., m, k]
 * B:   [..., k, n]
 * out: [..., m, n]
 */
Tensor TensorMultiply(Tensor A, Tensor B);

/*
 * Apply each filter across the height and width of each input. Each filter
 * produces one output feature map.
 *
 * input  = [batch size, input channels, height, width]
 * filter = [number of filters, input channels, filter height, filter width]
 * output = [batch size, number of filters, output height, output width]
 *
 * output height = (input height + 2*padding - filter height) / stride + 1
 * output width  = (input width  + 2*padding - filter width)  / stride + 1
 */
Tensor TensorConvolution(Tensor input, Tensor filter, int stride, int padding);

static inline Tensor TensorCopy(Tensor in)              { return TensorMap(IDENTITY, in); }
static inline Tensor TensorAdd(Tensor A, Tensor B)      { return TensorZip(ADD,      A, B); }
static inline Tensor TensorSubtract(Tensor A, Tensor B) { return TensorZip(SUBTRACT, A, B); }
static inline Tensor TensorHadamard(Tensor A, Tensor B) { return TensorZip(MULTIPLY, A, B); }


/*========================================================================
 *                                Utilities
 *========================================================================*/


/*
 * Returns a new tensor with rows shuffled via Fisher-Yates.
 * The same seed always produces the same permutation, allowing multiple
 * tensors to be shuffled in sync.
 */
Tensor TensorShuffle(Tensor A, unsigned int seed);

/*
 * Multiplies all entries of A by scalar in-place.
 */
void TensorScale(Tensor A, float scalar);

/*
 * Sets all entries of A to the given value.
 */
void TensorFill(Tensor A, Entry value);

/*
 * Xavier initialize a tensor using the last two dimensions as fan_in/fan_out.
 */
void TensorXavier(Tensor A);

/*
 * Converts an integer label tensor of shape [...] into a one-hot tensor of
 * shape [..., numClasses]. Labels must be non-negative integers in [0, numClasses)
 */
Tensor TensorOneHot(Tensor labels, int numClasses);

/*
 * Prints a tensor to stdout. Tensors of size 1 are printed as scalars.
 */
void TensorPrint(Tensor A);

/*
 * Softmax over the last dimension.
 * out = exp(in / T) ⊘ sum(exp(in / T))
 *
 * in:  Input logits    [..., numClasses]
 * out: Probabilities   [..., numClasses]
 */
Tensor TensorSoftmax(Tensor in, float temperature);

/*
 * Log-softmax over the last dimension.
 *
 * in:  Input logits        [..., numClasses]
 * out: Log-probabilities   [..., numClasses]
 */
Tensor TensorLogSoftmax(Tensor in, float temperature);

/*
 * Negative log-likelihood over the last dimension.
 * Note: NLL(y, LogSoftmax(z)) ≡ CrossEntropy(y, softmax(z)), but leaving y_hat
 * in log space avoids floating point errors.
 * out = -sum(y ⊙ y_hat)
 *
 * y:     Target probability distribution        [..., numClasses]
 * y_hat: Predicted log-probability distribution [..., numClasses]
 * out:   Mean NLL loss                          [] (Scalar)
 */
Tensor TensorNLL(Tensor y, Tensor y_hat);

/*
 * Mean squared error across all entries.
 * out = mean((y - y_hat)^2)
 *
 * y:     Target values    [..., numClasses]
 * y_hat: Predicted values [..., numClasses]
 * out:   Mean MSE loss    [] (Scalar)
 */
Tensor TensorMSE(Tensor y, Tensor y_hat);

/*
 * Samples an index along the last dimension of logits for each row.
 * temperature == 0 is greedy, otherwise samples from softmax distribution.
 * Assumes contiguous row-major layout - do not use on transposed tensors.
 * logits: [..., numClasses]
 * out:    [...]
 */
Tensor TensorSample(Tensor logits, float temperature);

/*
 * Returns the number of positions at which corresponding entries of A and B differ.
 */
int TensorHammingDistance(Tensor A, Tensor B);

/*
 * Starts with loss tensor, takes mean for each entry, then propogates gradients
 * backwards through all recorded operations. All gradients are zero'd first and
 * then accumulated during the backward pass.
 */
void TensorBackward(Tensor loss);

/*
 * Stochastic gradient descent with optional momentum. Set momentum to 0 for
 * standard SGD. Must be called after TensorBackward.
 *
 * momentum == 0: A -= lr * gradients.
 * momentum != 0: m = momentum * m + gradients; A -= lr * m.
 */
void TensorSGD(Tensor A, float lr, float momentum);

/*
 * Adam optimizer with optional weight decay. Set weightDecay to 0 for Adam
 * and weightDecay > 0 for AdamW. Must be called after TensorBackward.
 *
 * m = beta1 * m + (1 - beta1) * gradients
 * v = beta2 * v + (1 - beta2) * gradients^2
 * A -= lr * (m_hat / (sqrt(v_hat) + epsilon) + weightDecay * A)
 */
void TensorAdam(Tensor A, float lr, float beta1, float beta2, float epsilon, float weightDecay);

#endif
