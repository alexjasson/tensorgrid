#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "tensorgrid.h"

/*
 * Autograd wrapper functions for the primitive tensor operations. Each function allocates
 * the output tensor, executes the operation and records it on the autograd tape.
 */
Tensor autogradMap(Unary f, Tensor A);
Tensor autogradZip(Binary f, Tensor A, Tensor B);
Tensor autogradMultiply(Tensor A, Tensor B);
Tensor autogradLinear(Tensor in, Tensor weight, Tensor bias, Unary f);
Tensor autogradConvolution(Tensor in, Tensor filter, Tensor bias, int stride, int padding, Unary f);
Tensor autogradPool(Binary f, Entry initial, Tensor in, int size, int stride, int padding);
Tensor autogradGather(Tensor A, Tensor B, int axis);
Tensor autogradScatter(Binary f, Tensor A, Tensor B, Entry fill, int axis, int n);
Tensor autogradUnfold(Tensor A, Entry fill, int fH, int fW, int stride, int padding);

/*
 * Starts with loss tensor, takes mean for each entry, then propagates gradients
 * backwards through all recorded operations. All gradients are zero'd first and
 * then accumulated during the backward pass.
 */
void autogradBackward(Tensor loss);

#endif
