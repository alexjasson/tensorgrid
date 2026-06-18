// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include "MemoryPool.h"

#define POOL_SIZE 256

struct memoryPool {
    struct { void *ptr; size_t bytes; } slots[POOL_SIZE];
    int count;
};

static MemoryPool newMemoryPool(void)
{
    return calloc(1, sizeof(struct memoryPool));
}

static void freeMemoryPool(void)
{
    MemoryPool pool = MemoryPoolGet();
    for (int i = 0; i < pool->count; i++)
        free(pool->slots[i].ptr);
    free(pool);
}


void *MemoryPoolAlloc(MemoryPool pool, size_t bytes)
{
    for (int i = 0; i < pool->count; i++)
        if (pool->slots[i].bytes == bytes) {
            void *ptr = pool->slots[i].ptr;
            pool->slots[i] = pool->slots[--pool->count];
            return ptr;
        }
    return malloc(bytes);
}

void MemoryPoolFree(MemoryPool pool, void *ptr, size_t bytes)
{
    if (!ptr) return;
    if (pool->count < POOL_SIZE) {
        pool->slots[pool->count].ptr   = ptr;
        pool->slots[pool->count].bytes = bytes;
        pool->count++;
    } else {
        free(ptr);
    }
}

MemoryPool MemoryPoolGet(void)
{
    static MemoryPool pool;
    if (!pool) {
        pool = newMemoryPool();
        atexit(freeMemoryPool);
    }
    return pool;
}
