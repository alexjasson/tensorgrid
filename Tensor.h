#ifndef TENSOR_H
#define TENSOR_H

#define MAX_DIMENSIONS 8

typedef float Entry;                     // Change data type of entry as needed
typedef Entry (*UnaryFn)(Entry);         // z = f(x) on an entry
typedef Entry (*BinaryFn)(Entry, Entry); // z = f(x,y) on an entry

// Tensor representation
typedef struct {
    int dimensions;
    int shape[MAX_DIMENSIONS];
    int strides[MAX_DIMENSIONS];
    Entry *entries;
} Tensor;


/*========================================================================
 *                         Memory allocation
 *========================================================================*/

/*
 * Allocates a new tensor with all entries as 0. To avoid memory 
 * fragmentation, don't call this multiple times. Instead, allocate 
 * memory for entries separately and use TensorView.
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


/*========================================================================
 *                         Shape, views and acess
 *========================================================================*/

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
void TensorMap(Tensor out, UnaryFn f, Tensor A);

/*
 * out = f(A, B).
 */
void TensorZip(Tensor out, BinaryFn f, Tensor A, Tensor B);

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
 * out = in.
 */
void TensorCopy(Tensor out, Tensor in);


/*========================================================================
 *                                Utilities
 *========================================================================*/

/*
 * Xavier initialize a tensor using the last two dimensions as fan_in/fan_out.
 */
void TensorXavier(Tensor A);

/*
 * Sets all entries to zero.
 */
void TensorZero(Tensor A);

#endif
