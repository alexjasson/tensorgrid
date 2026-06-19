#include <stdint.h>
#include <float.h>
#include "Scalar.h"

// exp(a) with ~1e-5 relative error
Scalar expe(Scalar a)
{
    if (a >  88.0f) return FLT_MAX;
    if (a < -87.3f) return 0.0f;

    int   n = (int)(a * 1.4426950408889634f + (a >= 0.0f ? 0.5f : -0.5f));
    float r = a - (float)n * 0.6931471805599453f;

    // 5th-order Taylor: 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120
    float p = 1.0f + r + r * r * (0.5f + r * ((1.0f/6.0f) + r * ((1.0f/24.0f) + r * (1.0f/120.0f))));

    union { float f; int32_t i; } u;
    u.i = (127 + n) << 23;
    return p * u.f;
}

// log(a) with ~1e-5 relative error
Scalar loge(Scalar a)
{
    if (a <= 0.0f) return -FLT_MAX;

    union { float f; int32_t i; } u;
    u.f     = a;
    int   e = ((u.i >> 23) & 0xFF) - 127;
    u.i     = (u.i & 0x007FFFFF) | 0x3F800000;
    float m = u.f;

    float t  = (m - 1.0f) / (m + 1.0f);
    float t2 = t * t;
    float l  = 2.0f * t * (1.0f + t2 * ((1.0f/3.0f) + t2 * ((1.0f/5.0f) + t2 * (1.0f/7.0f))));

    return l + (float)e * 0.6931471805599453f;
}

// Unary functions and derivatives
Scalar relu(Scalar a)     { return a > 0.0f ? a : 0.0f; }
Scalar drelu(Scalar a)    { return a > 0.0f ? 1.0f : 0.0f; }
Scalar sigmoid(Scalar a)  { return 1.0f / (1.0f + expe(-a)); }
Scalar dsigmoid(Scalar a) { Scalar s = sigmoid(a); return s * (1.0f - s); }
Scalar dloge(Scalar a)    { return 1.0f / a; }
Scalar negate(Scalar a)   { return -a; }
Scalar identity(Scalar a) { return a; }
Scalar one(Scalar a)      { (void)a; return  1.0f; }
Scalar minusOne(Scalar a) { (void)a; return -1.0f; }
Scalar tanhe(Scalar a)    { return 2.0f / (1.0f + expe(-2.0f * a)) - 1.0f; }
Scalar dtanhe(Scalar a)   { Scalar t = tanhe(a); return 1.0f - t * t; }
Scalar sqrte(Scalar a)    { return __builtin_sqrtf(a); }
Scalar dsqrte(Scalar a)   { return 0.5f / sqrte(a); }
Scalar gelu(Scalar a)     { return 0.5f * a * (1.0f + tanhe(sqrte(2.0f / PI) * (a + 0.044715f * a * a * a))); }
Scalar dgelu(Scalar a)
{
    float u = tanhe(sqrte(2.0f / PI) * (a + 0.044715f * a * a * a));
    return 0.5f * (1.0f + u) + 0.5f * a * (1.0f - u * u) * sqrte(2.0f / PI) * (1.0f + 0.134145f * a * a);
}

// Binary functions and partial derivatives
Scalar add(Scalar a, Scalar b)       { return a + b; }
Scalar subtract(Scalar a, Scalar b)  { return a - b; }
Scalar multiply(Scalar a, Scalar b)  { return a * b; }
Scalar divide(Scalar a, Scalar b)    { return a / b; }
Scalar max(Scalar a, Scalar b)       { return a > b ? a : b; }
Scalar first(Scalar a, Scalar b)     { (void)b; return a; }
Scalar second(Scalar a, Scalar b)    { (void)a; return b; }
Scalar ones(Scalar a, Scalar b)      { (void)a; (void)b; return  1.0f; }
Scalar minusOnes(Scalar a, Scalar b) { (void)a; (void)b; return -1.0f; }
Scalar divideDa(Scalar a, Scalar b)  { (void)a; return 1.0f / b; }
Scalar divideDb(Scalar a, Scalar b)  { return -a / (b * b); }
Scalar maxDa(Scalar a, Scalar b)     { return a >= b ? 1.0f : 0.0f; }
Scalar maxDb(Scalar a, Scalar b)     { return b > a ? 1.0f : 0.0f; }
Scalar min(Scalar a, Scalar b)       { return a < b ? a : b; }
Scalar minDa(Scalar a, Scalar b)     { return a <= b ? 1.0f : 0.0f; }
Scalar minDb(Scalar a, Scalar b)     { return b < a ? 1.0f : 0.0f; }
