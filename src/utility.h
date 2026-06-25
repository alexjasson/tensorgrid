#ifndef UTILITY_H
#define UTILITY_H

float powi(float base, int exp);
float randomFloat(float lower, float upper);
int unravel(int i, const int *shape, const int *strides, int dims);
int isContiguous(const int *shape, const int *strides, int dims);

#endif
