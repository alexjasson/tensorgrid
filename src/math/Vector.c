#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "Vector.h"
#include "Scalar.h"


Vector VectorLoad(const void *p, DataType d)
{
    Vector x = VectorZero();
    switch (d) {
        case UINT8:   { const uint8_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case INT8:    { const int8_t  *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case INT16:   { const int16_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case INT32:   { const int32_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case FLOAT32: { const float   *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case FLOAT64: { const double  *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case FLOAT16:
#if defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)
            if (sizeof(Real) == 4) {
                __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
                memcpy(&x, &v, sizeof(x));
                break;
            }
#endif
            for (int i = 0; i < VECTOR_DIMS; i++)
                x[i] = ScalarLoad((const uint16_t *)p + i, FLOAT16);
            break;
        case REAL: memcpy(&x, p, sizeof(x)); break;
    }
    return x;
}
void   VectorStore(Real *p, Vector x)             { memcpy(p, &x, sizeof(x)); }
void   VectorFetch(const Real *p)                 { _mm_prefetch((const char *)p, _MM_HINT_ET0); }
Vector VectorZero(void)                           { return (Vector){0}; }
Vector VectorBcast(Real x)                        { return VectorZero() + x; }
Vector VectorAdd(Vector a, Vector b)              { return a + b; }
Vector VectorSub(Vector a, Vector b)              { return a - b; }
Vector VectorMul(Vector a, Vector b)              { return a * b; }
Vector VectorDiv(Vector a, Vector b)              { return a / b; }
Vector VectorFma(Vector a, Vector b, Vector c)    { return a * b + c; }
Vector VectorFnmadd(Vector a, Vector b, Vector c) { return c - a * b; }

Vector VectorMax(Vector a, Vector b)
{
    for (int i = 0; i < VECTOR_DIMS; i++)
        a[i] = a[i] > b[i] ? a[i] : b[i];
    return a;
}

Vector VectorMin(Vector a, Vector b)
{
    for (int i = 0; i < VECTOR_DIMS; i++)
        a[i] = a[i] < b[i] ? a[i] : b[i];
    return a;
}

Vector VectorGt0(Vector v)
{
    for (int i = 0; i < VECTOR_DIMS; i++)
        v[i] = v[i] > 0.0 ? 1.0 : 0.0;
    return v;
}

Vector VectorSqrt(Vector x)
{
    for (int i = 0; i < VECTOR_DIMS; i++)
        x[i] = sizeof(Real) == 8 ? __builtin_sqrt(x[i]) : __builtin_sqrtf(x[i]);
    return x;
}

Vector VectorRound(Vector x)
{
    for (int i = 0; i < VECTOR_DIMS; i++)
        x[i] = sizeof(Real) == 8 ? __builtin_rint(x[i]) : __builtin_rintf(x[i]);
    return x;
}

// 2^n for integer-valued n
Vector VectorPow2i(Vector n)
{
    for (int i = 0; i < VECTOR_DIMS; i++) {
        if (sizeof(Real) == 8) {
            union { double f; int64_t i; } u = { .i = (1023 + (int64_t)n[i]) << 52 };
            n[i] = u.f;
        } else {
            union { float f; int32_t i; } u = { .i = (127 + (int32_t)n[i]) << 23 };
            n[i] = u.f;
        }
    }
    return n;
}

// The unbiased exponent of x
Vector VectorLog2i(Vector x)
{
    for (int i = 0; i < VECTOR_DIMS; i++) {
        if (sizeof(Real) == 8) {
            union { double f; int64_t i; } u = { .f = x[i] };
            x[i] = (Real)(((u.i >> 52) & 0x7FF) - 1023);
        } else {
            union { float f; int32_t i; } u = { .f = x[i] };
            x[i] = (Real)(((u.i >> 23) & 0xFF) - 127);
        }
    }
    return x;
}

// The mantissa of x, as a Vector of Reals in [1, 2)
Vector VectorMantissa(Vector x)
{
    for (int i = 0; i < VECTOR_DIMS; i++) {
        if (sizeof(Real) == 8) {
            union { double f; int64_t i; } u = { .f = x[i] };
            u.i = (u.i & 0x000FFFFFFFFFFFFFLL) | 0x3FF0000000000000LL;
            x[i] = u.f;
        } else {
            union { float f; int32_t i; } u = { .f = x[i] };
            u.i = (u.i & 0x007FFFFF) | 0x3F800000;
            x[i] = u.f;
        }
    }
    return x;
}

Real VectorHsum(Vector v)
{
    Real s = 0.0;
    for (int i = 0; i < VECTOR_DIMS; i++) s += v[i];
    return s;
}

// exp(a) with ~1e-6 (float) or ~1e-13 (double) relative error
Vector expv(Vector a)
{
    Real expMax = sizeof(Real) == 8 ?  709.0 :  88.0;
    Real expMin = sizeof(Real) == 8 ? -708.0 : -87.3;
    a = VectorMin(VectorMax(a, VectorBcast(expMin)), VectorBcast(expMax));
    Vector n = VectorRound(VectorMul(a, VectorBcast(1.4426950408889634)));
    Vector r = VectorFnmadd(n, VectorBcast(0.6931471805599453), a);
    Vector p = VectorBcast(1.0 / 120.0);
    if (sizeof(Real) == 8) {
        p = VectorBcast(1.0 / 6227020800.0);                 // 1/13!
        p = VectorFma(p, r, VectorBcast(1.0 / 479001600.0)); // 1/12!
        p = VectorFma(p, r, VectorBcast(1.0 / 39916800.0));  // 1/11!
        p = VectorFma(p, r, VectorBcast(1.0 / 3628800.0));   // 1/10!
        p = VectorFma(p, r, VectorBcast(1.0 / 362880.0));    // 1/9!
        p = VectorFma(p, r, VectorBcast(1.0 / 40320.0));     // 1/8!
        p = VectorFma(p, r, VectorBcast(1.0 / 5040.0));      // 1/7!
        p = VectorFma(p, r, VectorBcast(1.0 / 720.0));       // 1/6!
        p = VectorFma(p, r, VectorBcast(1.0 / 120.0));
    }
    p = VectorFma(p, r, VectorBcast(1.0 / 24.0));
    p = VectorFma(p, r, VectorBcast(1.0 / 6.0));
    p = VectorFma(p, r, VectorBcast(0.5));
    p = VectorFma(p, r, VectorBcast(1.0));
    p = VectorFma(p, r, VectorBcast(1.0));
    return VectorMul(p, VectorPow2i(n));
}

// log(a) with ~1e-5 (float) or ~1e-11 (double) relative error
Vector logv(Vector a)
{
    Vector e = VectorLog2i(a);
    Vector m = VectorMantissa(a);
    Vector t = VectorDiv(VectorSub(m, VectorBcast(1.0)), VectorAdd(m, VectorBcast(1.0)));
    Vector t2 = VectorMul(t, t);
    Vector p = VectorBcast(1.0 / 7.0);
    if (sizeof(Real) == 8) {
        p = VectorBcast(1.0 / 19.0);
        p = VectorFma(p, t2, VectorBcast(1.0 / 17.0));
        p = VectorFma(p, t2, VectorBcast(1.0 / 15.0));
        p = VectorFma(p, t2, VectorBcast(1.0 / 13.0));
        p = VectorFma(p, t2, VectorBcast(1.0 / 11.0));
        p = VectorFma(p, t2, VectorBcast(1.0 / 9.0));
        p = VectorFma(p, t2, VectorBcast(1.0 / 7.0));
    }
    p = VectorFma(p, t2, VectorBcast(1.0 / 5.0));
    p = VectorFma(p, t2, VectorBcast(1.0 / 3.0));
    p = VectorFma(p, t2, VectorBcast(1.0));
    Vector l = VectorMul(VectorMul(VectorBcast(2.0), t), p);
    return VectorFma(e, VectorBcast(0.6931471805599453), l);  // l + e*ln2
}

// Unary functions and derivatives for vectors
Vector reluv(Vector a)      { return VectorMax(a, VectorZero()); }
Vector dreluv(Vector a)     { return VectorGt0(a); }
Vector identityv(Vector a)  { return a; }
Vector didentityv(Vector a) { (void)a; return VectorBcast(1.0); }
Vector negatev(Vector a)    { return VectorSub(VectorZero(), a); }
Vector dnegatev(Vector a)   { (void)a; return VectorBcast(-1.0); }
Vector sigmoidv(Vector a)   { return VectorDiv(VectorBcast(1.0), VectorAdd(VectorBcast(1.0), expv(VectorSub(VectorZero(), a)))); }
Vector dsigmoidv(Vector a)  { Vector s = sigmoidv(a); return VectorMul(s, VectorSub(VectorBcast(1.0), s)); }
Vector tanhv(Vector a)      { return VectorSub(VectorDiv(VectorBcast(2.0), VectorAdd(VectorBcast(1.0), expv(VectorMul(VectorBcast(-2.0), a)))), VectorBcast(1.0)); }
Vector dtanhv(Vector a)     { Vector t = tanhv(a); return VectorSub(VectorBcast(1.0), VectorMul(t, t)); }
Vector dlogv(Vector a)      { return VectorDiv(VectorBcast(1.0), a); }
Vector sqrtv(Vector a)      { return VectorSqrt(a); }
Vector dsqrtv(Vector a)     { return VectorDiv(VectorBcast(0.5), VectorSqrt(a)); }
static Vector geluArg(Vector a) { return VectorMul(VectorBcast(0.7978845608028654), VectorAdd(a, VectorMul(VectorBcast(0.044715), VectorMul(a, VectorMul(a, a))))); } // 0.797... = sqrt(2/pi)
Vector geluv(Vector a)      { return VectorMul(VectorBcast(0.5), VectorMul(a, VectorAdd(VectorBcast(1.0), tanhv(geluArg(a))))); }
Vector dgeluv(Vector a)
{
    Vector c = VectorBcast(0.7978845608028654); // sqrt(2/pi)
    Vector u = tanhv(geluArg(a));
    Vector t1 = VectorMul(VectorBcast(0.5), VectorAdd(VectorBcast(1.0), u));
    Vector t2 = VectorMul(VectorMul(VectorMul(VectorBcast(0.5), a), VectorSub(VectorBcast(1.0), VectorMul(u, u))),
                VectorMul(c, VectorAdd(VectorBcast(1.0), VectorMul(VectorBcast(0.134145), VectorMul(a, a)))));
    return VectorAdd(t1, t2);
}

// Binary functions and partial derivatives for vectors
Vector addv(Vector a, Vector b)       { return VectorAdd(a, b); }
Vector subtractv(Vector a, Vector b)  { return VectorSub(a, b); }
Vector multiplyv(Vector a, Vector b)  { return VectorMul(a, b); }
Vector dividev(Vector a, Vector b)    { return VectorDiv(a, b); }
Vector maxv(Vector a, Vector b)       { return VectorMax(a, b); }
Vector firstv(Vector a, Vector b)     { (void)b; return a; }
Vector secondv(Vector a, Vector b)    { (void)a; return b; }
Vector onesv(Vector a, Vector b)      { (void)a; (void)b; return VectorBcast(1.0); }
Vector minusOnesv(Vector a, Vector b) { (void)a; (void)b; return VectorBcast(-1.0); }
Vector divideDav(Vector a, Vector b)  { (void)a; return VectorDiv(VectorBcast(1.0), b); }
Vector divideDbv(Vector a, Vector b)  { return VectorSub(VectorZero(), VectorDiv(a, VectorMul(b, b))); }
Vector maxDav(Vector a, Vector b)     { return VectorSub(VectorBcast(1.0), VectorGt0(VectorSub(b, a))); }
Vector maxDbv(Vector a, Vector b)     { return VectorGt0(VectorSub(b, a)); }
Vector minv(Vector a, Vector b)       { return VectorMin(a, b); }
Vector minDav(Vector a, Vector b)     { return VectorSub(VectorBcast(1.0), VectorGt0(VectorSub(a, b))); }
Vector minDbv(Vector a, Vector b)     { return VectorGt0(VectorSub(a, b)); }
