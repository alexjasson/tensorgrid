#ifndef OPERATION_H
#define OPERATION_H

#include "tensorgrid.h"

// Represents an operation applied to one or two input Tensors, producing an output Tensor
typedef struct {
    Tensor out, A, B;
    enum { Map, Zip, Multiply, Unfold, Gather, Scatter } type;
    union { Univariate u; Bivariate b; };
    int   axis;                    // Reduce axis (Gather, Scatter)
    Entry initial;                 // Identity value (Scatter, Unfold)
    int   fH, fW, stride, padding; // Unfold arguments
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
Tensor OperationGather(Tensor A, int axis, Tensor indices);
Tensor OperationScatter(Bivariate f, Entry initial, Tensor A, int axis, Tensor indices, int n);
Tensor OperationUnfold(Tensor input, int fH, int fW, int stride, int padding, Entry fill);

/*
 * Starts with loss tensor, takes mean for each entry, then propagates gradients
 * backwards through all recorded operations. All gradients are zero'd first and
 * then accumulated during the backward pass.
 */
void OperationBackward(Tensor loss);

#endif
