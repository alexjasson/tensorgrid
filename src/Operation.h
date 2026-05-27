#ifndef OPERATION_H
#define OPERATION_H

#include "tensorgrid.h"

// Represents a primitive operation on a tensor
typedef struct {
    Tensor out, A, B;
    enum { Map, Zip, Multiply, Affine, Linear, Unfold, Gather, Scatter } type;
    union { Univariate u; Bivariate b; };
    Entry fill;
    int axis, fH, fW, stride, padding, inner;
    Tensor bias, product;
} Operation;

void OperationForward(Operation op);
void OperationBackward(Operation op);

#endif
