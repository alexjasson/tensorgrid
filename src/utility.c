// https://github.com/alexjasson/tensorgrid

#include <stdint.h>
#include <float.h>
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

// exp(x) with ~1e-5 relative error
float fastExp(float x)
{
    if (x >  88.7f) return FLT_MAX;
    if (x < -87.3f) return 0.0f;

    int   n = (int)(x * 1.4426950408889634f + (x >= 0.0f ? 0.5f : -0.5f));
    float r = x - (float)n * 0.6931471805599453f;

    // 5th-order Taylor: 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    float p = 1.0f + r + r * r * (0.5f + r * ((1.0f/6.0f) + r * ((1.0f/24.0f) + r * (1.0f/120.0f))));

    union { float f; int32_t i; } u;
    u.i = (127 + n) << 23;
    return p * u.f;
}

// log(x) with ~1e-5 relative error
float fastLog(float x)
{
    if (x <= 0.0f) return -FLT_MAX;

    union { float f; int32_t i; } u;
    u.f     = x;
    int   e = ((u.i >> 23) & 0xFF) - 127;
    u.i     = (u.i & 0x007FFFFF) | 0x3F800000;
    float m = u.f;

    float t  = (m - 1.0f) / (m + 1.0f);
    float t2 = t * t;
    float l  = 2.0f * t * (1.0f + t2 * ((1.0f/3.0f) + t2 * ((1.0f/5.0f) + t2 * (1.0f/7.0f))));

    return l + (float)e * 0.6931471805599453f;
}

float fastPow(float base, int exp)
{
    float result = 1.0f;
    while (exp > 0) {
        if (exp & 1) result *= base;
        base *= base;
        exp >>= 1;
    }
    return result;
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
