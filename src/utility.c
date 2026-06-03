// https://github.com/alexjasson/tensorgrid

#include <stdint.h>
#include <time.h>
#include "utility.h"

float randomFloat(float lower, float upper)
{
    static uint32_t state = 0;
    static int seeded = 0;

    if (!seeded) {
        state = (uint32_t)time(NULL);
        seeded = 1;
    }

    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;

    float r = (state >> 8) / 16777216.0f;
    return lower + r * (upper - lower);
}

int unravel(int i, const int *shape, const int *strides, int dims)
{
    int offset = 0;
    for (int d = dims - 1; d >= 0; d--) {
        int s = shape[d];
        offset += (i % s) * strides[d];
        i /= s;
    }
    return offset;
}

int isContiguous(const int *shape, const int *strides, int dims)
{
    int expected = 1;
    for (int d = dims - 1; d >= 0; d--) {
        if (strides[d] != expected) return 0;
        expected *= shape[d];
    }
    return 1;
}
