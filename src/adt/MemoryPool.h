#ifndef MEMORYPOOL_H
#define MEMORYPOOL_H

#include <stddef.h>

typedef struct memoryPool *MemoryPool;

/*
 * Singleton pattern to initialize a memory pool, created on first use and destroyed at exit.
 */
MemoryPool MemoryPoolGet(void);

/*
 * Returns a pointer to a zeroed block of memory of size bytes. Reuses memory from
 * pool when possible. Returns NULL if allocation fails.
 */
void *MemoryPoolAlloc(MemoryPool pool, size_t bytes);

/*
 * Recycles memory into pool for later use. If the pool is full the memory is freed.
 */
void  MemoryPoolFree(MemoryPool pool, void *ptr, size_t bytes);

#endif
