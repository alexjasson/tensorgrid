#ifndef OPERATION_H
#define OPERATION_H

#include "tensorgrid.h"

typedef struct {
    Tensor out, A, B;
    enum { Map, Zip, Multiply, Unfold, Gather, Scatter } type;
    union { Univariate u; Bivariate b; };
    Entry fill;
    int axis, fH, fW, stride, padding;
} Operation;

void OperationForward(Operation op);
void OperationBackward(Operation op);

#endif
