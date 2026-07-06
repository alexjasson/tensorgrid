#ifndef MATRIX_H
#define MATRIX_H

#include "../tensorgrid.h"

/*
 * A strided 2D matrix view over existing memory. Element (i, j) is
 * data[i*sRow + j*sCol].
 */
typedef struct { Real *data; int rows, cols, sRow, sCol; } Matrix;

/*
 * Returns a strided matrix view over existing memory.
 */
Matrix MatrixView(Real *data, int rows, int cols, int sRow, int sCol);

/*
 * A and B are multiplied and the result is added to C.
 * C += A @ B.
 */
void MatrixMultiply(Matrix A, Matrix B, Matrix C);

#endif
