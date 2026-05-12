#ifndef TENSORGRID_H
#define TENSORGRID_H

#define MAX_DIMENSIONS 8
#define TRUE 1
#define FALSE 0
#define INFINITE 1e+38

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

// Univariate/Bivariate functions to be applied to tensors - add more as needed
extern Univariate RELU;
extern Univariate SIGMOID;
extern Univariate LOG;
extern Univariate EXP;
extern Univariate NEGATE;
extern Univariate IDENTITY;
extern Univariate TANH;
extern Bivariate  ADD;
extern Bivariate  SUBTRACT;
extern Bivariate  MULTIPLY;
extern Bivariate  DIVIDE;
extern Bivariate  MAX;


/*=============================================================================================
 *                                      Memory management
 *=============================================================================================*/

/*
 * Allocates a new tensor with all entries as 0. If autograd is nonzero,
 * the tensor also allocates gradient storage and optimizer state.
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


/*=============================================================================================
 *                                  Shape, views and access
 *=============================================================================================*/

/*
 * Returns the total number of entries in the tensor.
 */
int TensorSize(Tensor A);

/*
 * Returns the size of the tensor along the given axis.
 */
int TensorShape(Tensor A, int axis);

/*
 * Returns a view of A with autograd removed. Shares memory with A.
 * Operations on the returned tensor are no longer recorded.
 */
Tensor TensorDetach(Tensor A);

/*
 * Returns a view of length entries starting along axis 0. Shares memory with A.
 */
Tensor TensorSlice(Tensor A, int index, int length);

/*
 * Returns a view of A with a new shape. Total size must be unchanged. Shares memory with A.
 */
Tensor TensorReshape(Tensor A, int dimensions, int *shape);

/*
 * Returns a new tensor of the given shape and strides. Allocates new memory.
 */
Tensor TensorRestride(Tensor A, int dimensions, int *shape, int *strides);

/*
 * Returns a new tensor with the last two dimensions transposed. Allocates new memory.
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


/*=============================================================================================
 *                                   Composite Operations
 *=============================================================================================*/

/*
 * Embedding lookup. Each integer index selects a row of the embedding table.
 * table:   [numClasses, embeddingSize]
 * indices: [...]                       (integer values in [0, numClasses), encoded as Entry)
 * out:     [..., embeddingSize]
 */
Tensor TensorEmbedding(Tensor table, Tensor indices);

/*
 * Dense layer with optional activation function. Pass IDENTITY for no activation.
 * out = f(in @ weight + bias)
 */
Tensor TensorLinearLayer(Tensor in, Tensor weight, Tensor bias, Univariate f);

/*
 * Convolution layer with optional activation function. Pass IDENTITY for no activation.
 * out = f(conv(in, filter) + bias)
 */
Tensor TensorConvolutionLayer(Tensor in, Tensor filter, Tensor bias, int stride, int padding, Univariate f);

/*
 * Max-pooling over the spatial dimensions.
 */
Tensor TensorMaxPool(Tensor input, int size, int stride, int padding);

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
 * Cross-entropy loss over the last dimension.
 * out = NLL(y, LogSoftmax(logits))
 *
 * y:      Target probability distribution [..., numClasses]
 * logits: Predicted logits                [..., numClasses]
 * out:    Mean cross-entropy loss         [] (Scalar)
 */
Tensor TensorCrossEntropyLoss(Tensor y, Tensor logits, float temperature);

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

/*
 * Moves a size x size window across height and width, and reduces each window
 * with a bivariate function like max or sum.
 *
 * input  = [batch size, channels, height, width]
 * output = [batch size, channels, output height, output width]
 *
 * output height = (input height + 2*padding - size) / stride + 1
 * output width  = (input width  + 2*padding - size) / stride + 1
 */
Tensor TensorPool(Bivariate f, Entry initial, Tensor input, int size, int stride, int padding);

/*
 * Broadcasts the given axis from size 1 to size n; all entries are replicated
 * along that axis. Implemented as a Gather with a length-n index tensor of zeros.
 * A:   [..., 1, ...]
 * out: [..., n, ...]
 */
Tensor TensorBroadcast(Tensor A, int axis, int n);

/*
 * Reduces A along the given axis using f, starting from initial. Implemented as
 * a Scatter with a length-A.shape[axis] index tensor of zeros and n=1.
 * A:   [..., k, ...]
 * out: [..., 1, ...]
 */
Tensor TensorReduce(Bivariate f, Entry initial, Tensor A, int axis);

/*
 * Element-wise addition. out = A + B
 */
Tensor TensorAdd(Tensor A, Tensor B);

/*
 * Element-wise subtraction. out = A - B
 */
Tensor TensorSubtract(Tensor A, Tensor B);

/*
 * Element-wise (Hadamard) product. out = A * B
 */
Tensor TensorHadamard(Tensor A, Tensor B);

/*
 * Returns a new tensor with the same shape and values as A. Allocates new memory.
 */
Tensor TensorCopy(Tensor A);


/*=============================================================================================
 *                                   Primitive Operations
 *=============================================================================================*/

/*
 * out = f(A).
 */
Tensor TensorMap(Univariate f, Tensor A);

/*
 * out = f(A, B). Shapes must match, except axes of size 1, which are broadcasted
 * to the other tensor's size.
 */
Tensor TensorZip(Bivariate f, Tensor A, Tensor B);

/*
 * Multiply A by B using the last two dimensions as matrices.
 * A:   [..., m, k]
 * B:   [..., k, n]
 * out: [..., m, n]
 */
Tensor TensorMultiply(Tensor A, Tensor B);

/*
* Read from A at positions given by indices.
*
* A:       [..., s_0, ..., s_m, ...]   (indices shape lives at axis)
* indices: [s_0, ..., s_m]             (integer values in [0, n), encoded as Entry)
* out:     [..., n, ...]
*/
Tensor TensorGather(Tensor A, int axis, Tensor indices);

/*
 * Write values from A into positions given by indices.
 * If multiple values go to the same position, combine them with f.
 *
 * A:       [..., K, ...]               (K is the size of the gathered axis)
 * indices: [s_0, ..., s_m]             (integer values in [0, K), encoded as Entry)
 * out:     [..., s_0, ..., s_m, ...]
 */
Tensor TensorScatter(Bivariate f, Entry initial, Tensor A, int axis, Tensor indices, int n);

/*
 * Moves a fH x fW window across the height and width of each input, and flattens each window
 * into a column of the output. Out-of-bounds positions are filled with `fill`. Input must
 * be 4D.
 *
 * input  = [batch size, input channels, height, width]
 * output = [batch size, input channels * fH * fW, output height * output width]
 *
 * output height = (input height + 2*padding - fH) / stride + 1
 * output width  = (input width  + 2*padding - fW) / stride + 1
 */
Tensor TensorUnfold(Tensor input, int fH, int fW, int stride, int padding, Entry fill);


/*=============================================================================================
 *                                          Utilities
 *=============================================================================================*/

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
