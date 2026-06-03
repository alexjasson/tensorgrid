#include <stdint.h>
#include "Vector.h"

#define PI 3.141592653589793f

#if defined(__AVX512F__)

Vector VectorLoad(const float *p)                 { return _mm512_loadu_ps(p); }
void   VectorStore(float *p, Vector x)            { _mm512_storeu_ps(p, x); }
Vector VectorBcast(float x)                       { return _mm512_set1_ps(x); }
Vector VectorAdd(Vector a, Vector b)              { return _mm512_add_ps(a, b); }
Vector VectorSub(Vector a, Vector b)              { return _mm512_sub_ps(a, b); }
Vector VectorMul(Vector a, Vector b)              { return _mm512_mul_ps(a, b); }
Vector VectorDiv(Vector a, Vector b)              { return _mm512_div_ps(a, b); }
Vector VectorFma(Vector a, Vector b, Vector c)    { return _mm512_fmadd_ps(a, b, c); }
Vector VectorZero(void)                           { return _mm512_setzero_ps(); }
Vector VectorMax(Vector a, Vector b)              { return _mm512_max_ps(a, b); }
Vector VectorMin(Vector a, Vector b)              { return _mm512_min_ps(a, b); }
Vector VectorFnmadd(Vector a, Vector b, Vector c) { return _mm512_fnmadd_ps(a, b, c); }
Vector VectorRound(Vector x)    { return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
Vector VectorPow2i(Vector n)    { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(n), _mm512_set1_epi32(127)), 23)); }
Vector VectorSqrt(Vector x)     { return _mm512_sqrt_ps(x); }
Vector VectorLog2i(Vector x)    { return _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_and_epi32(_mm512_srli_epi32(_mm512_castps_si512(x), 23), _mm512_set1_epi32(0xFF)), _mm512_set1_epi32(127))); }
Vector VectorMantissa(Vector x) { return _mm512_castsi512_ps(_mm512_or_epi32(_mm512_and_epi32(_mm512_castps_si512(x), _mm512_set1_epi32(0x007FFFFF)), _mm512_set1_epi32(0x3F800000))); }
Vector VectorGt0(Vector v)      { return _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ), _mm512_set1_ps(1.0f)); }
float  VectorHsum(Vector v)     { return _mm512_reduce_add_ps(v); }

#elif defined(__AVX2__)

Vector VectorLoad(const float *p)                 { return _mm256_loadu_ps(p); }
void   VectorStore(float *p, Vector x)            { _mm256_storeu_ps(p, x); }
Vector VectorBcast(float x)                       { return _mm256_set1_ps(x); }
Vector VectorAdd(Vector a, Vector b)              { return _mm256_add_ps(a, b); }
Vector VectorSub(Vector a, Vector b)              { return _mm256_sub_ps(a, b); }
Vector VectorMul(Vector a, Vector b)              { return _mm256_mul_ps(a, b); }
Vector VectorDiv(Vector a, Vector b)              { return _mm256_div_ps(a, b); }
Vector VectorFma(Vector a, Vector b, Vector c)    { return _mm256_fmadd_ps(a, b, c); }
Vector VectorZero(void)                           { return _mm256_setzero_ps(); }
Vector VectorMax(Vector a, Vector b)              { return _mm256_max_ps(a, b); }
Vector VectorMin(Vector a, Vector b)              { return _mm256_min_ps(a, b); }
Vector VectorFnmadd(Vector a, Vector b, Vector c) { return _mm256_fnmadd_ps(a, b, c); }
Vector VectorRound(Vector x)    { return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
Vector VectorPow2i(Vector n)    { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)), 23)); }
Vector VectorSqrt(Vector x)     { return _mm256_sqrt_ps(x); }
Vector VectorLog2i(Vector x)    { return _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(x), 23), _mm256_set1_epi32(0xFF)), _mm256_set1_epi32(127))); }
Vector VectorMantissa(Vector x) { return _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000))); }
Vector VectorGt0(Vector v)      { return _mm256_and_ps(_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ), _mm256_set1_ps(1.0f)); }
float  VectorHsum(Vector v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

#else

Vector VectorLoad(const float *p)                 { return *p; }
void   VectorStore(float *p, Vector x)            { *p = x; }
Vector VectorBcast(float x)                       { return x; }
Vector VectorAdd(Vector a, Vector b)              { return a + b; }
Vector VectorSub(Vector a, Vector b)              { return a - b; }
Vector VectorMul(Vector a, Vector b)              { return a * b; }
Vector VectorDiv(Vector a, Vector b)              { return a / b; }
Vector VectorFma(Vector a, Vector b, Vector c)    { return a * b + c; }
Vector VectorZero(void)                           { return 0.0f; }
Vector VectorMax(Vector a, Vector b)              { return a > b ? a : b; }
Vector VectorMin(Vector a, Vector b)              { return a < b ? a : b; }
Vector VectorFnmadd(Vector a, Vector b, Vector c) { return c - a * b; }
Vector VectorRound(Vector x)                { return __builtin_rintf(x); }
Vector VectorPow2i(Vector n)                { return __builtin_ldexpf(1.0f, (int)n); }
Vector VectorSqrt(Vector x)                 { return __builtin_sqrtf(x); }
Vector VectorLog2i(Vector x)                { union { float f; int32_t i; } u = { .f = x }; return (Vector)(((u.i >> 23) & 0xFF) - 127); }
Vector VectorMantissa(Vector x)             { union { float f; int32_t i; } u = { .f = x }; u.i = (u.i & 0x007FFFFF) | 0x3F800000; return u.f; }
Vector VectorGt0(Vector v)                  { return v > 0.0f ? 1.0f : 0.0f; }
float  VectorHsum(Vector v)                 { return v; }

#endif

// exp(a) with ~1e-6 relative error
Vector expv(Vector a)
{
    a = VectorMin(VectorMax(a, VectorBcast(-87.3f)), VectorBcast(88.7f));
    Vector n = VectorRound(VectorMul(a, VectorBcast(1.4426950408889634f)));
    Vector r = VectorFnmadd(n, VectorBcast(0.6931471805599453f), a);
    Vector p = VectorBcast(1.0f / 120.0f);
    p = VectorFma(p, r, VectorBcast(1.0f / 24.0f));
    p = VectorFma(p, r, VectorBcast(1.0f / 6.0f));
    p = VectorFma(p, r, VectorBcast(0.5f));
    p = VectorFma(p, r, VectorBcast(1.0f));
    p = VectorFma(p, r, VectorBcast(1.0f));
    return VectorMul(p, VectorPow2i(n));
}

// log(a) with ~1e-5 relative error
Vector logv(Vector a)
{
    Vector e = VectorLog2i(a);
    Vector m = VectorMantissa(a);
    Vector t = VectorDiv(VectorSub(m, VectorBcast(1.0f)), VectorAdd(m, VectorBcast(1.0f)));
    Vector t2 = VectorMul(t, t);
    Vector p = VectorBcast(1.0f / 7.0f);
    p = VectorFma(p, t2, VectorBcast(1.0f / 5.0f));
    p = VectorFma(p, t2, VectorBcast(1.0f / 3.0f));
    p = VectorFma(p, t2, VectorBcast(1.0f));
    Vector l = VectorMul(VectorMul(VectorBcast(2.0f), t), p);
    return VectorFma(e, VectorBcast(0.6931471805599453f), l);  // l + e*ln2
}

// Unary functions and derivatives for vectors
Vector reluv(Vector a)      { return VectorMax(a, VectorZero()); }
Vector dreluv(Vector a)     { return VectorGt0(a); }
Vector identityv(Vector a)  { return a; }
Vector didentityv(Vector a) { (void)a; return VectorBcast(1.0f); }
Vector negatev(Vector a)    { return VectorSub(VectorZero(), a); }
Vector dnegatev(Vector a)   { (void)a; return VectorBcast(-1.0f); }
Vector sigmoidv(Vector a)   { return VectorDiv(VectorBcast(1.0f), VectorAdd(VectorBcast(1.0f), expv(VectorSub(VectorZero(), a)))); }
Vector dsigmoidv(Vector a)  { Vector s = sigmoidv(a); return VectorMul(s, VectorSub(VectorBcast(1.0f), s)); }
Vector tanhv(Vector a)      { return VectorSub(VectorDiv(VectorBcast(2.0f), VectorAdd(VectorBcast(1.0f), expv(VectorMul(VectorBcast(-2.0f), a)))), VectorBcast(1.0f)); }
Vector dtanhv(Vector a)     { Vector t = tanhv(a); return VectorSub(VectorBcast(1.0f), VectorMul(t, t)); }
Vector dlogv(Vector a)      { return VectorDiv(VectorBcast(1.0f), a); }
Vector sqrtv(Vector a)      { return VectorSqrt(a); }
Vector dsqrtv(Vector a)     { return VectorDiv(VectorBcast(0.5f), VectorSqrt(a)); }
static Vector geluArg(Vector a) { return VectorMul(VectorBcast(__builtin_sqrtf(2.0f / PI)), VectorAdd(a, VectorMul(VectorBcast(0.044715f), VectorMul(a, VectorMul(a, a))))); }
Vector geluv(Vector a)      { return VectorMul(VectorBcast(0.5f), VectorMul(a, VectorAdd(VectorBcast(1.0f), tanhv(geluArg(a))))); }
Vector dgeluv(Vector a)
{
    Vector c = VectorBcast(__builtin_sqrtf(2.0f / PI));
    Vector u = tanhv(geluArg(a));
    Vector t1 = VectorMul(VectorBcast(0.5f), VectorAdd(VectorBcast(1.0f), u));
    Vector t2 = VectorMul(VectorMul(VectorMul(VectorBcast(0.5f), a), VectorSub(VectorBcast(1.0f), VectorMul(u, u))),
                VectorMul(c, VectorAdd(VectorBcast(1.0f), VectorMul(VectorBcast(0.134145f), VectorMul(a, a)))));
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
Vector onesv(Vector a, Vector b)      { (void)a; (void)b; return VectorBcast(1.0f); }
Vector minusOnesv(Vector a, Vector b) { (void)a; (void)b; return VectorBcast(-1.0f); }
Vector divideDav(Vector a, Vector b)  { (void)a; return VectorDiv(VectorBcast(1.0f), b); }
Vector divideDbv(Vector a, Vector b)  { return VectorSub(VectorZero(), VectorDiv(a, VectorMul(b, b))); }
Vector maxDav(Vector a, Vector b)     { return VectorSub(VectorBcast(1.0f), VectorGt0(VectorSub(b, a))); }
Vector maxDbv(Vector a, Vector b)     { return VectorGt0(VectorSub(b, a)); }
Vector minv(Vector a, Vector b)       { return VectorMin(a, b); }
Vector minDav(Vector a, Vector b)     { return VectorSub(VectorBcast(1.0f), VectorGt0(VectorSub(a, b))); }
Vector minDbv(Vector a, Vector b)     { return VectorGt0(VectorSub(a, b)); }
