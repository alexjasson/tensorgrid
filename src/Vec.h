#ifndef VEC_H
#define VEC_H

#include <immintrin.h>
#include <stdint.h>

#if defined(__AVX512F__)

typedef __m512 Vec;
#define VEC_DIMS 16
static inline Vec   VecLoad(const float *p)        { return _mm512_loadu_ps(p); }
static inline void  VecStore(float *p, Vec x)      { _mm512_storeu_ps(p, x); }
static inline Vec   VecBcast(float x)              { return _mm512_set1_ps(x); }
static inline Vec   VecAdd(Vec a, Vec b)           { return _mm512_add_ps(a, b); }
static inline Vec   VecSub(Vec a, Vec b)           { return _mm512_sub_ps(a, b); }
static inline Vec   VecMul(Vec a, Vec b)           { return _mm512_mul_ps(a, b); }
static inline Vec   VecDiv(Vec a, Vec b)           { return _mm512_div_ps(a, b); }
static inline Vec   VecFma(Vec a, Vec b, Vec c)    { return _mm512_fmadd_ps(a, b, c); }
static inline Vec   VecZero(void)                  { return _mm512_setzero_ps(); }
static inline Vec   VecMax(Vec a, Vec b)           { return _mm512_max_ps(a, b); }
static inline Vec   VecMin(Vec a, Vec b)           { return _mm512_min_ps(a, b); }
static inline Vec   VecFnmadd(Vec a, Vec b, Vec c) { return _mm512_fnmadd_ps(a, b, c); }
static inline Vec   VecRound(Vec x)                { return _mm512_roundscale_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline Vec   VecPow2i(Vec n)                { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvttps_epi32(n), _mm512_set1_epi32(127)), 23)); }
static inline Vec   VecSqrt(Vec x)                 { return _mm512_sqrt_ps(x); }
static inline Vec   VecLog2i(Vec x)                { return _mm512_cvtepi32_ps(_mm512_sub_epi32(_mm512_and_epi32(_mm512_srli_epi32(_mm512_castps_si512(x), 23), _mm512_set1_epi32(0xFF)), _mm512_set1_epi32(127))); }
static inline Vec   VecMantissa(Vec x)             { return _mm512_castsi512_ps(_mm512_or_epi32(_mm512_and_epi32(_mm512_castps_si512(x), _mm512_set1_epi32(0x007FFFFF)), _mm512_set1_epi32(0x3F800000))); }
static inline Vec   VecGt0(Vec v)                  { return _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(v, _mm512_setzero_ps(), _CMP_GT_OQ), _mm512_set1_ps(1.0f)); }
static inline float VecHsum(Vec v)                 { return _mm512_reduce_add_ps(v); }

#elif defined(__AVX2__)

typedef __m256 Vec;
#define VEC_DIMS 8
static inline Vec   VecLoad(const float *p)        { return _mm256_loadu_ps(p); }
static inline void  VecStore(float *p, Vec x)      { _mm256_storeu_ps(p, x); }
static inline Vec   VecBcast(float x)              { return _mm256_set1_ps(x); }
static inline Vec   VecAdd(Vec a, Vec b)           { return _mm256_add_ps(a, b); }
static inline Vec   VecSub(Vec a, Vec b)           { return _mm256_sub_ps(a, b); }
static inline Vec   VecMul(Vec a, Vec b)           { return _mm256_mul_ps(a, b); }
static inline Vec   VecDiv(Vec a, Vec b)           { return _mm256_div_ps(a, b); }
static inline Vec   VecFma(Vec a, Vec b, Vec c)    { return _mm256_fmadd_ps(a, b, c); }
static inline Vec   VecZero(void)                  { return _mm256_setzero_ps(); }
static inline Vec   VecMax(Vec a, Vec b)           { return _mm256_max_ps(a, b); }
static inline Vec   VecMin(Vec a, Vec b)           { return _mm256_min_ps(a, b); }
static inline Vec   VecFnmadd(Vec a, Vec b, Vec c) { return _mm256_fnmadd_ps(a, b, c); }
static inline Vec   VecRound(Vec x)                { return _mm256_round_ps(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline Vec   VecPow2i(Vec n)                { return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(n), _mm256_set1_epi32(127)), 23)); }
static inline Vec   VecSqrt(Vec x)                 { return _mm256_sqrt_ps(x); }
static inline Vec   VecLog2i(Vec x)                { return _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_and_si256(_mm256_srli_epi32(_mm256_castps_si256(x), 23), _mm256_set1_epi32(0xFF)), _mm256_set1_epi32(127))); }
static inline Vec   VecMantissa(Vec x)             { return _mm256_castsi256_ps(_mm256_or_si256(_mm256_and_si256(_mm256_castps_si256(x), _mm256_set1_epi32(0x007FFFFF)), _mm256_set1_epi32(0x3F800000))); }
static inline Vec   VecGt0(Vec v)                  { return _mm256_and_ps(_mm256_cmp_ps(v, _mm256_setzero_ps(), _CMP_GT_OQ), _mm256_set1_ps(1.0f)); }
static inline float VecHsum(Vec v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

#else

typedef float Vec;
#define VEC_DIMS 1
static inline Vec   VecLoad(const float *p)        { return *p; }
static inline void  VecStore(float *p, Vec x)      { *p = x; }
static inline Vec   VecBcast(float x)              { return x; }
static inline Vec   VecAdd(Vec a, Vec b)           { return a + b; }
static inline Vec   VecSub(Vec a, Vec b)           { return a - b; }
static inline Vec   VecMul(Vec a, Vec b)           { return a * b; }
static inline Vec   VecDiv(Vec a, Vec b)           { return a / b; }
static inline Vec   VecFma(Vec a, Vec b, Vec c)    { return a * b + c; }
static inline Vec   VecZero(void)                  { return 0.0f; }
static inline Vec   VecMax(Vec a, Vec b)           { return a > b ? a : b; }
static inline Vec   VecMin(Vec a, Vec b)           { return a < b ? a : b; }
static inline Vec   VecFnmadd(Vec a, Vec b, Vec c) { return c - a * b; }
static inline Vec   VecRound(Vec x)                { return __builtin_rintf(x); }
static inline Vec   VecPow2i(Vec n)                { return __builtin_ldexpf(1.0f, (int)n); }
static inline Vec   VecSqrt(Vec x)                 { return __builtin_sqrtf(x); }
static inline Vec   VecLog2i(Vec x)                { union { float f; int32_t i; } u = { .f = x }; return (Vec)(((u.i >> 23) & 0xFF) - 127); }
static inline Vec   VecMantissa(Vec x)             { union { float f; int32_t i; } u = { .f = x }; u.i = (u.i & 0x007FFFFF) | 0x3F800000; return u.f; }
static inline Vec   VecGt0(Vec v)                  { return v > 0.0f ? 1.0f : 0.0f; }
static inline float VecHsum(Vec v)                 { return v; }

#endif

#endif
