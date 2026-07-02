#ifndef STACK_H
#define STACK_H

#include <stddef.h>

typedef struct stack *Stack; // LIFO stack, grows as needed

/*
 * Creates an empty stack holding elements of elementSize bytes each.
 */
Stack StackNew(size_t elementSize);

/*
 * Pushes element onto the top of the stack.
 */
void StackPush(Stack s, void *element);

/*
 * Removes the top element and copies it into out (may be NULL if no element).
 * Returns 1, or 0 if the stack was empty.
 */
int StackPop(Stack s, void *out);

/*
 * Returns 1 if the stack holds no elements.
 */
int StackEmpty(Stack s);

/*
 * Resets the stack to empty, keeping its capacity.
 */
void StackClear(Stack s);

/*
 * Frees the stack and its elements from memory.
 */
void StackFree(Stack s);

#endif
