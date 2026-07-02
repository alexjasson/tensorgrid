#ifndef HASHMAP_H
#define HASHMAP_H

#include <stddef.h>
#include <stdint.h>

typedef struct hashMap *HashMap; // Key (uint64) -> Value (long) map, grows as needed

/*
 * Creates an empty map with at least minCapacity slots.
 */
HashMap HashMapNew(size_t minCapacity);

/*
 * Frees the map.
 */
void HashMapFree(HashMap m);

/*
 * Resets the map to empty, keeping its capacity.
 */
void HashMapClear(HashMap m);

/*
 * Returns the value for key, or -1 if the key is absent.
 */
long HashMapGet(HashMap m, uint64_t key);

/*
 * Sets key's value, inserting or overwriting.
 */
void HashMapStore(HashMap m, uint64_t key, long value);

/*
 * Finds the key with the greatest value. Returns 1 and writes it to *key, otherwise
 * returns 0 if the map is empty. If there are multiple keys with the same value, the 
 * smallest key is returned.
 */
int HashMapArgmax(HashMap m, uint64_t *key);

#endif
