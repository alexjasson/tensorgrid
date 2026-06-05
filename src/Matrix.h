#ifndef MATRIX_H
#define MATRIX_H

#include "tensorgrid.h"

/*
 * A strided 2D matrix view over existing memory. Element (i, j) is 
 * data[i*sRow + j*sCol].
 */
typedef struct { Entry *data; int rows, cols, sRow, sCol; } Matrix;

/*
 * A and B are multiplied and the result is added to C.
 * C += A @ B. 
 */
void MatrixMultiply(Matrix A, Matrix B, Matrix C);

#endif
