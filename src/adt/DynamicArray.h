#ifndef DYNAMICARRAY_H
#define DYNAMICARRAY_H

#include <stddef.h>

typedef struct dynamicArray *DynamicArray;

/*
 * Creates an empty array holding elements of elementSize bytes each.
 */
DynamicArray DynamicArrayNew(size_t elementSize);

/*
 * Adds element to the end of the array, growing capacity as needed.
 */
void DynamicArrayPush(DynamicArray arr, void *element);

/*
 * Returns a pointer to the element at the given index.
 */
void *DynamicArrayGet(DynamicArray arr, int index);

/*
 * Returns the number of elements in the array.
 */
int DynamicArraySize(DynamicArray arr);

/*
 * Resets the array to empty.
 */
void DynamicArrayClear(DynamicArray arr);

/*
 * Frees the array and it's elements from memory.
 */
void DynamicArrayFree(DynamicArray arr);

#endif
