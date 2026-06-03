#ifndef UTILITY_H
#define UTILITY_H

float randomFloat(float lower, float upper);
float fastExp(float x);
float fastLog(float x);
float fastPow(float base, int exp);
int unravel(int i, const int *shape, const int *strides, int dims);
int isContiguous(const int *shape, const int *strides, int dims);

#endif
