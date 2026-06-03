#ifndef OPERATION_H
#define OPERATION_H

#include "tensorgrid.h"

// Scalar functions, move to Scalar.h later
typedef Entry (*ScalarUnary)(Entry);
typedef Entry (*ScalarBinary)(Entry, Entry);

// Represents a primitive operation on a tensor
typedef struct {
    Tensor out, A, B;
    enum { Map, Zip, Multiply, Affine, Linear, Convolution, Pool, Unfold, Gather, Scatter } type;
    union { Unary u; Binary b; };
    Entry fill;
    int axis, fH, fW, stride, padding, inner;
    Tensor bias, product, scratch;
} Operation;

void OperationForward(Operation op);
void OperationBackward(Operation op);

#endif
