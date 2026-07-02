#ifndef TENSORGRID_H
#define TENSORGRID_H

#define MAX_DIMENSIONS 8
#define TRUE 1
#define FALSE 0
#define INFINITE 1e+38
#define PI 3.141592653589793f

typedef void (*f)(void); // A function applied to scalars or vectors

/*
 * Unary/Binary functions y = f(a) and y = f(a, b), with scalar derivatives
 * dy/da (da) and dy/db (db). Vectorized equivalents are denoted with a v.
 */
typedef struct { f f, da;     f fv, dav;      } Unary;
typedef struct { f f, da, db; f fv, dav, dbv; } Binary;
extern Unary RELU, GELU, SIGMOID, LOG, EXP, NEGATE, IDENTITY, TANH, SQRT;
extern Binary ADD, SUBTRACT, MULTIPLY, DIVIDE, MAX, MIN;

typedef float Entry;
typedef struct optimizer Optimizer;
typedef enum { DataUint8, DataInt8, DataInt16, DataInt32, DataFloat32, DataFloat64 } DataType;

// Tensor representation
typedef struct {
    int dimensions;
    int shape[MAX_DIMENSIONS];
    int strides[MAX_DIMENSIONS];
    Entry *entries;
    Entry *gradients;     // Gradients for each entry, NULL if no autograd
    Optimizer *optimizer; // Optimizer state, NULL if no autograd
    int   *refcount;      // Number of references, NULL for views
} Tensor;

typedef struct tensorArray *TensorArray; // Array of tensors intended to be used for model parameters


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
 * Creates an empty tensor array.
 */
TensorArray TensorArrayNew(void);

/*
 * Allocates a new tensor, adds it to the array and returns it.
 */
Tensor TensorArrayAdd(TensorArray ta, int dimensions, int *shape, int autograd);

/*
 * Frees a tensor from memory.
 */
void TensorFree(Tensor A);

/*
 * Frees every tensor and the tensor array itself
 */
void TensorArrayFree(TensorArray ta);

/*
 * Reads a .tgd file containing one tensor and returns it.
 */
Tensor TensorRead(char *path);

/*
 * Reads raw binary data from a file and returns it as a tensor. Data starts
 * at offset bytes and is cast to Entry.
 */
Tensor TensorReadData(char *path, int dimensions, int *shape, long offset, DataType d);

/*
 * Writes a tensor to a .tgd file.
 */
void TensorWrite(Tensor A, char *path);

/*
 * Saves a tensor array to a .tgd file.
 */
void TensorArrayWrite(TensorArray ta, char *path);

/*
 * Given a tensor array, reads entries from a .tgd file into it. The tensor array
 * must have the same shape/number of tensors as the file. Returns 1 if successful
 * or 0 otherwise.
 */
int TensorArrayRead(TensorArray ta, char *path);


/*=============================================================================================
 *                                  Shape, views and access
 *=============================================================================================*/

/*
 * Returns the total number of entries in the tensor.
 */
int TensorSize(Tensor A);

/*
 * Adds the total number of entries in each tensor and returns it.
 */
int TensorArraySize(TensorArray ta);

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
 * Returns a view of A at the given index along axis 0, with axis 0 removed.
 * Result has one fewer dimension than A. Shares memory with A.
 */
Tensor TensorIndex(Tensor A, int index);

/*
 * Returns a view of A with a new shape. Total size must be unchanged and A must
 * be contiguous. Shares memory with A.
 */
Tensor TensorReshape(Tensor A, int dimensions, int *shape);

/*
 * Returns a view of A with the given shape and strides, starting at the given
 * element offset. Shares memory with A.
 */
Tensor TensorRestride(Tensor A, int dimensions, int *shape, int *strides, int offset);

/*
 * Returns a view of A with the last two dimensions transposed. Shares memory with A.
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
 *                                     Tensor operations
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
Tensor TensorLinearLayer(Tensor in, Tensor weight, Tensor bias, Unary fn);

/*
 * Convolution layer with optional activation function. Pass IDENTITY for no activation.
 * out = f(conv(in, filter) + bias)
 */
Tensor TensorConvolutionLayer(Tensor in, Tensor filter, Tensor bias, int stride, int padding, Unary fn);

/*
 * Max-pooling over the spatial dimensions.
 */
Tensor TensorMaxPool(Tensor input, int size, int stride, int padding);

/*
 * Layer normalization over the last dimension.
 * out = weight * (in - mean(in)) / sqrt(variance(in) + epsilon) + bias
 *
 * in:     [..., D]
 * weight: [..., D]
 * bias:   [..., D]
 * out:    [..., D]
 */
Tensor TensorLayerNorm(Tensor in, Tensor weight, Tensor bias, float eps);

/*
 * Returns an n x n causal mask: 0 on and below the diagonal, -INFINITE above.
 */
Tensor TensorCausalMask(int n);

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
 * y:      Target probability distribution [..., numClasses] or integer class labels [..., 1]
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
Tensor TensorPool(Binary fn, Entry initial, Tensor input, int size, int stride, int padding);

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
Tensor TensorReduce(Binary fn, Entry initial, Tensor A, int axis);

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

/*
 * out = f(A).
 */
Tensor TensorMap(Unary fn, Tensor A);

/*
 * out = f(A, B). Shapes are right-aligned (the input with fewer dimensions gains leading size-1
 * axes), then any axis of size 1 is broadcast to the other tensor's size along that axis.
 */
Tensor TensorZip(Binary fn, Tensor A, Tensor B);

/*
 * Multiply A by B using the last two dimensions as matrices.
 * A:   [..., m, k]
 * B:   [..., k, n]
 * out: [..., m, n]
 */
Tensor TensorMultiply(Tensor A, Tensor B);

/*
 * out = f(A + bias). Each bias entry is broadcast over the outer and inner axes.
 * A:    [outer, biasSize, inner]
 * bias: [biasSize]
 * out:  [outer, biasSize, inner]
 */
Tensor TensorAffine(Unary fn, Tensor A, Tensor bias, int inner);

/*
* Read from A at positions given by indices.
*
* A:       [..., s_0, ..., s_m, ...]   (indices shape lives at axis)
* indices: [s_0, ..., s_m]             (integer values in [0, n), encoded as Entry)
* out:     [..., n, ...]
*/
Tensor TensorGather(Tensor A, Tensor indices, int axis);

/*
 * Write values from A into positions given by indices.
 * If multiple values go to the same position, combine them with f.
 *
 * A:       [..., K, ...]               (K is the size of the gathered axis)
 * indices: [s_0, ..., s_m]             (integer values in [0, K), encoded as Entry)
 * out:     [..., s_0, ..., s_m, ...]
 */
Tensor TensorScatter(Binary fn, Tensor A, Tensor indices, Entry initial, int axis, int n);


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
 * Starts with loss tensor, takes mean for each entry, then propagates gradients
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
 * Applies a single SGD optimizer step to every tensor in the array.
 */
void TensorArraySGD(TensorArray ta, float lr, float momentum);

/*
 * Adam optimizer with optional weight decay. Set weightDecay to 0 for Adam
 * and weightDecay > 0 for AdamW. Must be called after TensorBackward.
 *
 * m = beta1 * m + (1 - beta1) * gradients
 * v = beta2 * v + (1 - beta2) * gradients^2
 * A -= lr * (m_hat / (sqrt(v_hat) + epsilon) + weightDecay * A)
 */
void TensorAdam(Tensor A, float lr, float beta1, float beta2, float epsilon, float weightDecay);

/*
 * Applies a single Adam optimizer step to every tensor in the array.
 */
void TensorArrayAdam(TensorArray ta, float lr, float beta1, float beta2, float epsilon, float weightDecay);

/*
 * Byte Pair Encoding algorithm for NLP
 * 
 * Train learns merge pairs from corpus in merges table
 * Encode encodes corpus into tokens using the trained merges table
 * Decode decodes tokens back into raw bytes using the trained merges table
 * 
 * corpus: [numBytes] tensor of raw byte values/characters in [0, 255]
 * tokens: [numTokens] tensor of token ids in [0, 256 + numMerges)
 * merges: [numMerges, 2] tensor of learned merge pairs
 */
void TensorBPETrain(Tensor corpus, Tensor merges);
Tensor TensorBPEEncode(Tensor corpus, Tensor merges);
Tensor TensorBPEDecode(Tensor tokens, Tensor merges);

#endif
