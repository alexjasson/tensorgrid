#include <stdlib.h>
#include <string.h>
#include "HashMap.h"

struct hashMap {
    uint64_t      *keys;
    long          *vals;
    unsigned char *used;
    size_t         capacity;
    size_t         size;
};

static uint64_t hash(uint64_t x);
static void insert(HashMap m, uint64_t key, long val);
static void grow(HashMap m);

static uint64_t hash(uint64_t x)
{
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

static void insert(HashMap m, uint64_t key, long val)
{
    size_t mask = m->capacity - 1;
    size_t i = hash(key) & mask;
    while (m->used[i]) i = (i + 1) & mask;
    m->used[i] = 1; m->keys[i] = key; m->vals[i] = val; m->size++;
}

static void grow(HashMap m)
{
    size_t         oldCapacity = m->capacity;
    uint64_t      *ok = m->keys;
    long          *ov = m->vals;
    unsigned char *ou = m->used;

    m->capacity   = oldCapacity * 2;
    m->keys  = malloc(m->capacity * sizeof *m->keys);
    m->vals  = malloc(m->capacity * sizeof *m->vals);
    m->used  = calloc(m->capacity, 1);
    m->size = 0;
    for (size_t i = 0; i < oldCapacity; i++)
        if (ou[i]) insert(m, ok[i], ov[i]);
    free(ok); free(ov); free(ou);
}

HashMap HashMapNew(size_t minCapacity)
{
    HashMap m = malloc(sizeof *m);

    if (minCapacity < 16) minCapacity = 16;
    m->capacity = 1;
    while (m->capacity < minCapacity) m->capacity <<= 1;

    m->keys  = malloc(m->capacity * sizeof *m->keys);
    m->vals  = malloc(m->capacity * sizeof *m->vals);
    m->used  = calloc(m->capacity, 1);
    m->size = 0;
    return m;
}

void HashMapFree(HashMap m)
{
    if (!m) return;
    free(m->keys); free(m->vals); free(m->used); free(m);
}

void HashMapClear(HashMap m)
{
    memset(m->used, 0, m->capacity);
    m->size = 0;
}

long HashMapGet(HashMap m, uint64_t key)
{
    size_t mask = m->capacity - 1;
    size_t i = hash(key) & mask;
    while (m->used[i]) {
        if (m->keys[i] == key) return m->vals[i];
        i = (i + 1) & mask;
    }
    return -1;
}

void HashMapStore(HashMap m, uint64_t key, long value)
{
    size_t mask = m->capacity - 1;
    size_t i = hash(key) & mask;
    while (m->used[i]) {
        if (m->keys[i] == key) { m->vals[i] = value; return; }
        i = (i + 1) & mask;
    }
    if (m->size * 10 >= m->capacity * 7) grow(m);
    insert(m, key, value);
}

int HashMapArgmax(HashMap m, uint64_t *key)
{
    long best = -1; uint64_t bestKey = 0; int found = 0;
    for (size_t s = 0; s < m->capacity; s++) {
        if (!m->used[s]) continue;
        long v = m->vals[s];
        if (!found || v > best || (v == best && m->keys[s] < bestKey)) {
            best = v; bestKey = m->keys[s]; found = 1;
        }
    }
    if (found) *key = bestKey;
    return found;
}
