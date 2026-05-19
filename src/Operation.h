#ifndef OPERATION_H
#define OPERATION_H

#include "tensorgrid.h"

// Represents an operation applied to one or two input Tensors, producing an output Tensor
typedef struct {
    Tensor out, A, B;
    enum { Map, Zip, Multiply, Unfold, Gather, Scatter } type;
    union { Univariate u; Bivariate b; };
    Entry fill;
    int   axis, fH, fW, stride, padding;
} Operation;

/*
 * Primitive operation constructors. Each allocates the output tensor and records the
 * operation internally if the output tensor requires autograd. Operations can then be
 * walked backwards during OperationBackward to accumulate all gradients for each input
 * Tensor.
 */
Tensor OperationMap(Univariate f, Tensor A);
Tensor OperationZip(Bivariate f, Tensor A, Tensor B);
Tensor OperationMultiply(Tensor A, Tensor B);
Tensor OperationGather(Tensor A, Tensor B, int axis);
Tensor OperationScatter(Bivariate f, Tensor A, Tensor B, Entry fill, int axis, int n);
Tensor OperationUnfold(Tensor A, Entry fill, int fH, int fW, int stride, int padding);

/*
 * Starts with loss tensor, takes mean for each entry, then propagates gradients
 * backwards through all recorded operations. All gradients are zero'd first and
 * then accumulated during the backward pass.
 */
void OperationBackward(Tensor loss);

#endif
