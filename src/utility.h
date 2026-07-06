#ifndef UTILITY_H
#define UTILITY_H

#include "tensorgrid.h"
#include <stdint.h>

Real powi(Real base, int exp);
uint64_t packi(int a, int b);
Real randomNumber(Real lower, Real upper);
int unravel(int i, const int *shape, const int *strides, int dims);
int isContiguous(const int *shape, const int *strides, int dims);

#endif
