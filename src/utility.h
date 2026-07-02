#ifndef UTILITY_H
#define UTILITY_H

#include <stdint.h>

float powi(float base, int exp);
uint64_t packi(int a, int b);
float randomFloat(float lower, float upper);
int unravel(int i, const int *shape, const int *strides, int dims);
int isContiguous(const int *shape, const int *strides, int dims);

#endif
