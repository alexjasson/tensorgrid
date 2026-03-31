#ifndef DYNAMICARRAY_H
#define DYNAMICARRAY_H

#include <stddef.h>

typedef struct dynamicArray *DynamicArray;

DynamicArray DynamicArrayNew(size_t elementSize);
void DynamicArrayPush(DynamicArray arr, void *element);
void *DynamicArrayGet(DynamicArray arr, int index);
int DynamicArraySize(DynamicArray arr);
void DynamicArrayClear(DynamicArray arr);
void DynamicArrayFree(DynamicArray arr);

#endif
