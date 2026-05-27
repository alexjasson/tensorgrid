#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensorgrid.h"

/*
 * Autograd wrapper functions for the primitive tensor operations. Each function allocates
 * the output tensor, executes the operation and records it on the autograd tape.
 */
Tensor autogradMap(Univariate f, Tensor A);
Tensor autogradZip(Bivariate f, Tensor A, Tensor B);
Tensor autogradMultiply(Tensor A, Tensor B);
Tensor autogradAffine(Univariate f, Tensor product, Tensor bias, int inner);
Tensor autogradLinear(Tensor in, Tensor weight, Tensor bias, Univariate f);
Tensor autogradGather(Tensor A, Tensor B, int axis);
Tensor autogradScatter(Bivariate f, Tensor A, Tensor B, Entry fill, int axis, int n);
Tensor autogradUnfold(Tensor A, Entry fill, int fH, int fW, int stride, int padding);

/*
 * Starts with loss tensor, takes mean for each entry, then propagates gradients
 * backwards through all recorded operations. All gradients are zero'd first and
 * then accumulated during the backward pass.
 */
void autogradBackward(Tensor loss);

#endif
