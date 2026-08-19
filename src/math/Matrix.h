#ifndef MATRIX_H
#define MATRIX_H

#include "../tensorgrid.h"
#include "Scale.h"

/*
 * A strided 2D matrix view over existing memory. Element (i, j) is
 * factor * scale.values[i] * (data[i*sRow + j*sCol] - scale.zeros[i])
 */
typedef struct { Real *data; int rows, cols, sRow, sCol; DataType type; Real factor; Scale scale; } Matrix;

/*
 * Returns a strided matrix view over existing memory, stored as FLOAT32
 */
Matrix MatrixView(Real *data, int rows, int cols, int sRow, int sCol);

/*
 * A and B are multiplied and the result is added to C.
 * C += A @ B.
 * C is stored as FLOAT32
 */
void MatrixMultiply(Matrix A, Matrix B, Matrix C);

#endif
