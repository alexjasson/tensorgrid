#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include "../tensorgrid.h"
#include "Scale.h"

/*
 * A strided 2D matrix view over existing memory. Element (i, j) is
 * factor * scale.values[g] * (data[off + i*sRow + j*sCol] - scale.zeros[g]), where g is
 * the group covering the position (i, j) in the matrix. off counts entries, not bytes,
 * since sub-byte entries share a byte.
 */
typedef struct { Real *data; size_t off; int rows, cols, sRow, sCol; DataType type; Real factor; Scale scale; } Matrix;

/*
 * Returns a strided matrix view over existing memory, stored as type and read through
 * scale. A FLOAT32 view takes an empty scale.
 */
Matrix MatrixView(void *data, int rows, int cols, int sRow, int sCol, DataType type, Scale scale);

/*
 * A and B are multiplied and the result is added to C.
 * C += A @ B.
 * C is stored as FLOAT32
 */
void MatrixMultiply(Matrix A, Matrix B, Matrix C);

#endif
