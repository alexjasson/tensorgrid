#ifndef TENSOR_H
#define TENSOR_H

#define MAX_DIMENSIONS 8

typedef float Entry;                   // Change data type of entry as needed
typedef Entry (*EntryFunction)(Entry); // Single variable function on an entry

// Tensor representation
typedef struct {
    int dimensions;
    int shape[MAX_DIMENSIONS];
    int strides[MAX_DIMENSIONS];
    Entry *entries;
} Tensor;

/*
 * Allocates a new tensor on the heap with all entries as 0.
 * To avoid memory fragmentation, don't call this multiple times.
 * Instead, allocate memory for entries separately and use TensorView.
 */
Tensor TensorNew(int dimensions, int *shape);

/*
 * Frees a tensor from memory.
 */
void TensorFree(Tensor A);

/*
 * Creates a tensor view over existing memory.
 */
Tensor TensorView(int dimensions, int *shape, Entry *entries);

/*
 * Returns the total number of entries in the tensor
 */
int TensorSize(Tensor A);

/*
 * Returns a view of A[index] along the first axis (dimensions reduced by 1).
 * Shares memory with A, use with caution.
 */
Tensor TensorSlice(Tensor A, int index);

/*
 * Returns a view with a new shape. Total size must be unchanged.
 */
Tensor TensorReshape(Tensor A, int dimensions, int *shape);

/*
 * Returns a view with the last two dimensions transposed.
 */
Tensor TensorTranspose(Tensor A);

/*
 * out = A + B.
 */
void TensorAdd(Tensor out, Tensor A, Tensor B);

/*
 * out = A - B.
 */
void TensorSubtract(Tensor out, Tensor A, Tensor B);

/*
 * out = A ⊙ B.
 */
void TensorHadamard(Tensor out, Tensor A, Tensor B);

/*
 * Multiply A by B using A's last two dimensions as matrices.
 * A:   [..., m, k]
 * B:   [k, n]
 * out: [..., m, n]
 */
void TensorMultiply(Tensor out, Tensor A, Tensor B);

/*
 * Xavier initialize a tensor using the last two dimensions as fan_in/fan_out.
 */
void TensorXavier(Tensor A);

/*
 * Applies a function to each entry in the tensor.
 */
void TensorApply(Tensor A, EntryFunction f);

/*
 * out = in.
 */
void TensorCopy(Tensor out, Tensor in);

/*
 * Sets all entries to zero.
 */
void TensorZero(Tensor A);

/*
 * Returns the entry at the given N-dimensional index.
 */
Entry TensorGet(Tensor A, int *indices);

/*
 * Sets the entry at the given N-dimensional index.
 */
void TensorSet(Tensor A, int *indices, Entry value);

#endif
