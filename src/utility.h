#ifndef UTILITY_H
#define UTILITY_H

#include "Vec.h"

int min(int a, int b);
float randomFloat(float lower, float upper);
float fastExp(float x);
float fastLog(float x);
float fastPow(float base, int exp);
Vec   VecExp(Vec x);
Vec   VecLog(Vec x);
int unravel(int i, const int *shape, const int *strides, int dims);
int isContiguous(const int *shape, const int *strides, int dims);

#endif
