// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "DynamicArray.h"

#define START_CAPACITY 8

struct dynamicArray {
    void  *data;
    int    size;
    int    capacity;
    size_t elementSize;
};

DynamicArray DynamicArrayNew(size_t elementSize)
{
    DynamicArray arr = malloc(sizeof(*arr));
    if (!arr) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    arr->data = malloc(START_CAPACITY * elementSize);
    if (!arr->data) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    arr->size        = 0;
    arr->capacity    = START_CAPACITY;
    arr->elementSize = elementSize;
    return arr;
}

void DynamicArrayPush(DynamicArray arr, void *element)
{
    if (arr->size == arr->capacity) {
        arr->capacity *= 2;
        arr->data = realloc(arr->data, arr->capacity * arr->elementSize);
        if (!arr->data) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    memcpy((char *)arr->data + arr->size * arr->elementSize, element, arr->elementSize);
    arr->size++;
}

void *DynamicArrayGet(DynamicArray arr, int index)
{
    return (char *)arr->data + index * arr->elementSize;
}

int DynamicArraySize(DynamicArray arr)
{
    return arr->size;
}

void DynamicArrayClear(DynamicArray arr)
{
    arr->size = 0;
}

void DynamicArrayFree(DynamicArray arr)
{
    free(arr->data);
    free(arr);
}
