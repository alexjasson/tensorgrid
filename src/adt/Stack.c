// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "Stack.h"

#define START_CAPACITY 8

struct stack {
    void  *data;
    int    size;
    int    capacity;
    size_t elementSize;
};

Stack StackNew(size_t elementSize)
{
    Stack s = malloc(sizeof(*s));
    if (!s) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    s->data = malloc(START_CAPACITY * elementSize);
    if (!s->data) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    s->size        = 0;
    s->capacity    = START_CAPACITY;
    s->elementSize = elementSize;
    return s;
}

void StackPush(Stack s, void *element)
{
    if (s->size == s->capacity) {
        s->capacity *= 2;
        s->data = realloc(s->data, s->capacity * s->elementSize);
        if (!s->data) {
            fprintf(stderr, "Insufficient memory!\n");
            exit(EXIT_FAILURE);
        }
    }
    memcpy((char *)s->data + s->size * s->elementSize, element, s->elementSize);
    s->size++;
}

int StackPop(Stack s, void *out)
{
    if (s->size == 0) return 0;
    s->size--;
    if (out) memcpy(out, (char *)s->data + s->size * s->elementSize, s->elementSize);
    return 1;
}

int StackEmpty(Stack s)
{
    return s->size == 0;
}

void StackClear(Stack s)
{
    s->size = 0;
}

void StackFree(Stack s)
{
    free(s->data);
    free(s);
}
