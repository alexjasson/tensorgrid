#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "Vector.h"

typedef uint32_t Bits __attribute__((__vector_size__(VECTOR_DIMS * sizeof(uint32_t))));
typedef int32_t Signed __attribute__((__vector_size__(VECTOR_DIMS * sizeof(int32_t))));

static Real in(Real x, Real lo, Real hi) { return x < lo ? lo : x > hi ? hi : x; }
static Real nearest(Real x) { return x < 0 ? x - (Real)0.5 : x + (Real)0.5; }

// float16 -> Real
static Real halfToReal(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e    = (h >> 10) & 0x1Fu;
    uint32_t m    = h & 0x03FFu;
    union { uint32_t u; float f; } out;

    if (e == 31)     out.u = sign | 0x7F800000u | (m << 13);
    else if (e > 0)  out.u = sign | ((e - 15 + 127) << 23) | (m << 13);
    else if (m == 0) out.u = sign;
    else {
        int shift = 0;
        while (!(m & 0x0400u)) { m <<= 1; shift++; }
        out.u = sign | ((uint32_t)(127 - 15 + 1 - shift) << 23) | ((m & 0x03FFu) << 13);
    }
    return out.f;
}

// Real -> float16
static uint16_t realToHalf(Real x)
{
    union { float f; uint32_t u; } in = { .f = (float)x };
    uint16_t sign = (uint16_t)((in.u >> 16) & 0x8000u);
    uint32_t abs  = in.u & 0x7FFFFFFFu;
    int      e    = (int)(abs >> 23) - 127 + 15;
    uint32_t m    = abs & 0x007FFFFFu;

    if (abs > 0x7F800000u) return sign | 0x7E00u;
    if (e >= 31)           return sign | 0x7BFFu;
    if (e <= 0) {
        if (e < -10) return sign;
        m |= 0x00800000u;
        int shift = 14 - e;
        uint16_t h = sign | (uint16_t)(m >> shift);
        uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        return h + (rem > half || (rem == half && (h & 1)));
    }
    uint16_t h = sign | (uint16_t)(e << 10) | (uint16_t)(m >> 13);
    h += (m & 0x1FFFu) > 0x1000u || ((m & 0x1FFFu) == 0x1000u && (h & 1));
    if ((uint16_t)(h & 0x7FFFu) >= 0x7C00u) return sign | 0x7BFFu;
    return h;
}

static size_t dataBits(DataType d)
{
    switch (d) {
        case INT2:    return 2;
        case INT4:    return 4;
        case UINT8:
        case INT8:    return 8;
        case INT16:
        case FLOAT16: return 16;
        case INT32:
        case FLOAT32: return 32;
        case FLOAT64: return 64;
    }
    return 8;
}

// Size of n entries of the data type in bytes
size_t DataSize(DataType d, size_t n) { return (n * dataBits(d) + 7) / 8; }
int DataPerByte(DataType d) { return d == INT4 ? 2 : d == INT2 ? 4 : 1; }

int64_t DataMax(DataType d)
{
    switch (d) {
        case UINT8: return UINT8_MAX;
        case INT8:  return INT8_MAX;
        case INT16: return INT16_MAX;
        case INT32: return INT32_MAX;
        case INT4:  return 7;
        case INT2:  return 1;
        default:    return 0;
    }
}

int64_t DataMin(DataType d)
{
    switch (d) {
        case UINT8: return 0;
        case INT8:  return INT8_MIN;
        case INT16: return INT16_MIN;
        case INT32: return INT32_MIN;
        case INT4:  return -8;
        case INT2:  return -1; // ternary
        default:    return 0;
    }
}

static Real dataHigh(DataType d)
{
    Real h = (Real)DataMax(d);
    return (int64_t)h > DataMax(d) ? h * (Real)(1.0 - 1.0 / 16777216.0) : h;
}

static inline Vector bitsUniform(Bits z)
{
    z ^= z >> 16; z *= 0x7feb352du;
    z ^= z >> 15; z *= 0x846ca68bu;
    z ^= z >> 16;
    return __builtin_convertvector((Signed)(z >> 8), Vector) * (Real)(1.0 / 16777216.0);
}

static inline Vector uniformAt(uint32_t key, size_t index)
{
    Bits offset, high = (Bits){0} + (uint32_t)(index >> 32);
    for (int i = 0; i < VECTOR_DIMS; i++) offset[i] = (uint32_t)i;
    if ((index + VECTOR_DIMS - 1) >> 32 != index >> 32) // the indices straddle a multiple of 2^32
        for (int i = 0; i < VECTOR_DIMS; i++) high[i] = (uint32_t)((index + (size_t)i) >> 32);
    return bitsUniform(key ^ ((uint32_t)index + offset) * 0x85ebca6bu ^ high);
}

Vector VectorUniformAt(uint32_t key, size_t index) { return uniformAt(key, index); }

static int subGet(const void *base, size_t index, DataType d)
{
    const unsigned char *bytes = base;
    if (d == INT2) { int v = (bytes[index >> 2] >> (index & 3) * 2) & 3; return v > 1 ? v - 4 : v; }
    int v = (bytes[index >> 1] >> (index & 1) * 4) & 15;
    return v > 7 ? v - 16 : v;
}

static void subSet(void *base, size_t index, DataType d, int v)
{
    int shift = d == INT2 ? (int)(index & 3) * 2 : (int)(index & 1) * 4;
    unsigned char *byte = (unsigned char *)base + (d == INT2 ? index >> 2 : index >> 1);
    unsigned char mask = (unsigned char)((d == INT2 ? 3 : 15) << shift);
    *byte = (unsigned char)((*byte & ~mask) | (((unsigned)v << shift) & mask));
}

static int subClamp(DataType d, Real x) { return (int)in(x, (Real)DataMin(d), (Real)DataMax(d)); }

static Real byteGet(const void *p, DataType d)
{
    switch (d) {
        case UINT8:   return (Real)*(const unsigned char *)p;
        case INT8:    return (Real)*(const signed char *)p;
        case INT16:   { int16_t  x; memcpy(&x, p, 2); return (Real)x; }
        case INT32:   { int32_t  x; memcpy(&x, p, 4); return (Real)x; }
        case FLOAT16: { uint16_t x; memcpy(&x, p, 2); return halfToReal(x); }
        case FLOAT32: { float    x; memcpy(&x, p, 4); return (Real)x; }
        case FLOAT64: { double   x; memcpy(&x, p, 8); return (Real)x; }
        case INT4: case INT2: break; // narrower than a byte
    }
    return 0.0;
}

static void byteSet(void *p, DataType d, Real x)
{
    switch (d) {
        case UINT8:   *(unsigned char *)p = (unsigned char)nearest(in(x, 0.0, 255.0)); break;
        case INT8:    *(signed char *)p = (signed char)nearest(in(x, -128.0, 127.0)); break;
        case INT16:   { int16_t  v = (int16_t)nearest(in(x, -32768.0, 32767.0));           memcpy(p, &v, 2); break; }
        case INT32:   { int32_t  v = (int32_t)nearest(in(x, -2147483648.0, 2147483520.0)); memcpy(p, &v, 4); break; }
        case FLOAT16: { uint16_t v = realToHalf(x);                                        memcpy(p, &v, 2); break; }
        // Storing into the native width is exact, so it must not clamp away an infinity
        case FLOAT32: { float    v = sizeof(Real) == 4 ? (float)x
                                   : (float)in(x, -3.402823466e+38, 3.402823466e+38);      memcpy(p, &v, 4); break; }
        case FLOAT64: { double   v = (double)x;                                            memcpy(p, &v, 8); break; }
        case INT4: case INT2: break; // narrower than a byte
    }
}

Real DataGet(const void *base, size_t index, DataType d)
{
    if (DataPerByte(d) > 1) return (Real)subGet(base, index, d);
    return byteGet((const unsigned char *)base + DataSize(d, index), d);
}

void DataSet(void *base, size_t index, DataType d, Real x)
{
    if (DataPerByte(d) > 1) { subSet(base, index, d, subClamp(d, nearest(x))); return; }
    byteSet((unsigned char *)base + DataSize(d, index), d, x);
}

static inline __attribute__((always_inline)) Vector load(const void *p, DataType d)
{
    Vector x = VectorZero();
#if defined(__AVX2__) && !defined(__AVX512F__)
    if (sizeof(Real) == 4 && VECTOR_DIMS == 8 && (d == UINT8 || d == INT8 || d == INT16 || d == INT32)) {
        __m256i v = d == UINT8 ? _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)p))
                  : d == INT8  ? _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p))
                  : d == INT16 ? _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)p))
                  :              _mm256_loadu_si256((const __m256i *)p);
        __m256 f = _mm256_cvtepi32_ps(v);
        memcpy(&x, &f, sizeof(x));
        return x;
    }
#endif
    switch (d) {
        case UINT8:   { const uint8_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case INT8:    { const int8_t  *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case INT4: {
            const unsigned char *e = p;
#if defined(__AVX2__) && !defined(__AVX512F__)
            if (sizeof(Real) == 4 && VECTOR_DIMS == 8) {
                uint32_t w;
                memcpy(&w, e, 4);
                __m256i v = _mm256_srlv_epi32(_mm256_set1_epi32((int)w),
                                              _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28));
                v = _mm256_and_si256(v, _mm256_set1_epi32(0xF));
                v = _mm256_sub_epi32(_mm256_xor_si256(v, _mm256_set1_epi32(8)), _mm256_set1_epi32(8));
                __m256 f = _mm256_cvtepi32_ps(v);
                memcpy(&x, &f, sizeof(x));
                break;
            }
#endif
            for (int i = 0; i < VECTOR_DIMS; i++) {
                int v = (i & 1) ? e[i / 2] >> 4 : e[i / 2] & 0xF;
                x[i] = (Real)(v > 7 ? v - 16 : v);
            }
            break;
        }
        case INT2: {
            const unsigned char *e = p;
#if defined(__AVX2__) && !defined(__AVX512F__)
            if (sizeof(Real) == 4 && VECTOR_DIMS == 8) {
                uint16_t w;
                memcpy(&w, e, 2);
                __m256i v = _mm256_srlv_epi32(_mm256_set1_epi32((int)w),
                                              _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14));
                v = _mm256_and_si256(v, _mm256_set1_epi32(3));
                v = _mm256_sub_epi32(_mm256_xor_si256(v, _mm256_set1_epi32(2)), _mm256_set1_epi32(2));
                __m256 f = _mm256_cvtepi32_ps(v);
                memcpy(&x, &f, sizeof(x));
                break;
            }
#endif
            for (int i = 0; i < VECTOR_DIMS; i++) {
                int v = (e[i / 4] >> (i % 4) * 2) & 3;
                x[i] = (Real)(v > 1 ? v - 4 : v);
            }
            break;
        }
        case INT16:   { const int16_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case INT32:   { const int32_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        // The native width is the hot path: a whole vector is one copy, no conversion
        case FLOAT32:
            if (sizeof(Real) == 4) { memcpy(&x, p, sizeof(x)); break; }
            { const float  *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; }
            break;
        case FLOAT64:
            if (sizeof(Real) == 8) { memcpy(&x, p, sizeof(x)); break; }
            { const double *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; }
            break;
        case FLOAT16:
#if defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)
            if (sizeof(Real) == 4) {
                __m256 v = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
                memcpy(&x, &v, sizeof(x));
                break;
            }
#endif
            for (int i = 0; i < VECTOR_DIMS; i++)
                x[i] = byteGet((const uint16_t *)p + i, FLOAT16);
            break;
    }
    return x;
}

Vector VectorLoad(const void *p, DataType d)      { return load(p, d); }
void   VectorStore(Real *p, Vector x)             { memcpy(p, &x, sizeof(x)); }

Vector VectorLoadPartial(const Real *p, int n)
{
    Vector x = VectorZero();
    if (n == VECTOR_DIMS) memcpy(&x, p, sizeof(x));
    else for (int i = 0; i < n; i++) x[i] = p[i];
    return x;
}

void VectorStorePartial(Real *p, Vector x, int n)
{
    if (n == VECTOR_DIMS) memcpy(p, &x, sizeof(x));
    else for (int i = 0; i < n; i++) p[i] = x[i];
}

static int narrowPacked(void *p, DataType d, Vector x)
{
#if defined(__AVX2__) && !defined(__AVX512F__)
    if (sizeof(Real) == 4 && (d == INT8 || d == UINT8 || d == INT16)) {
        __m256i w;
        memcpy(&w, &x, sizeof(w));
        __m256i v = _mm256_cvttps_epi32(_mm256_castsi256_ps(w));
        __m128i n = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        if (d == INT8)       _mm_storel_epi64((__m128i *)p, _mm_packs_epi16(n, n));
        else if (d == UINT8) _mm_storel_epi64((__m128i *)p, _mm_packus_epi16(n, n));
        else                 _mm_storeu_si128((__m128i *)p, n);
        return 1;
    }
#endif
    (void)p; (void)d; (void)x;
    return 0;
}

// u rounds stochastically (x + u, floored), NULL to nearest
static inline __attribute__((always_inline)) void storeRounded(void *p, DataType d, Vector x, const Vector *u)
{
    if (!DataMax(d)) {
        if (d == FLOAT32 && sizeof(Real) == 4) { memcpy(p, &x, sizeof(x)); return; }
#if defined(__AVX2__) && defined(__F16C__) && !defined(__AVX512F__)
        if (d == FLOAT16 && sizeof(Real) == 4) { // as realToHalf: saturates at 65504, NaN stays NaN
            __m256 v, top = _mm256_set1_ps(65504.0f);
            memcpy(&v, &x, sizeof(v));
            // NaN found on the bits: -Ofast assumes floats are finite
            __m256i bits = _mm256_and_si256(_mm256_castps_si256(v), _mm256_set1_epi32(0x7FFFFFFF));
            __m256 nan = _mm256_castsi256_ps(_mm256_cmpgt_epi32(bits, _mm256_set1_epi32(0x7F800000)));
            __m256 clamped = _mm256_max_ps(_mm256_min_ps(v, top), _mm256_sub_ps(_mm256_setzero_ps(), top));
            v = _mm256_blendv_ps(clamped, v, nan);
            _mm_storeu_si128((__m128i *)p, _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
            return;
        }
#endif
        for (int i = 0; i < VECTOR_DIMS; i++)
            DataSet(p, (size_t)i, d, x[i]);
        return;
    }

    Vector lo = VectorBcast((Real)DataMin(d)), hi = VectorBcast(dataHigh(d));
    x = VectorMin(VectorMax(x, lo), hi);
    x = VectorMin(VectorMax(VectorFloor(x + (u ? *u : VectorBcast((Real)0.5))), lo), hi);

    if (narrowPacked(p, d, x)) return;
    switch (d) {
        case UINT8: { uint8_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (uint8_t)x[i]; break; }
        case INT8:  { int8_t  *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (int8_t)x[i];  break; }
        case INT16: { int16_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (int16_t)x[i]; break; }
        case INT4:  {
            unsigned char *e = p;
#if defined(__AVX2__) && !defined(__AVX512F__)
            if (sizeof(Real) == 4 && VECTOR_DIMS == 8) { // shift each nibble into place, OR them in pairs
                __m256 xs;
                memcpy(&xs, &x, sizeof(xs));
                __m256i v = _mm256_and_si256(_mm256_cvttps_epi32(xs), _mm256_set1_epi32(0xF));
                v = _mm256_sllv_epi32(v, _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4));
                v = _mm256_or_si256(v, _mm256_srli_epi64(v, 32));        // byte k of the result in element 2k
                v = _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
                __m128i n = _mm256_castsi256_si128(v);
                __m128i b = _mm_packus_epi16(_mm_packus_epi32(n, n), _mm_setzero_si128());
                int32_t w = _mm_cvtsi128_si32(b);
                memcpy(e, &w, 4);
                break;
            }
#endif
            int i = 0;
            for (; i + 1 < VECTOR_DIMS; i += 2)
                e[i / 2] = (unsigned char)(((int)x[i] & 0xF) | ((int)x[i + 1] & 0xF) << 4);
            if (i < VECTOR_DIMS)
                e[i / 2] = (unsigned char)((e[i / 2] & 0xF0) | ((int)x[i] & 0xF));
            break;
        }
        case INT2:  { // VECTOR_DIMS is a multiple of 4, so whole bytes
            unsigned char *e = p;
#if defined(__AVX2__) && !defined(__AVX512F__)
            if (sizeof(Real) == 4 && VECTOR_DIMS == 8) { // shift each crumb into place, OR them in fours
                __m256 xs;
                memcpy(&xs, &x, sizeof(xs));
                __m256i v = _mm256_and_si256(_mm256_cvttps_epi32(xs), _mm256_set1_epi32(3));
                v = _mm256_sllv_epi32(v, _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6));
                v = _mm256_or_si256(v, _mm256_srli_epi64(v, 32));
                v = _mm256_or_si256(v, _mm256_srli_si256(v, 8));          // byte k of the result in element 4k
                uint16_t w = (uint16_t)(_mm256_extract_epi32(v, 0) | _mm256_extract_epi32(v, 4) << 8);
                memcpy(e, &w, 2);
                break;
            }
#endif
            for (int i = 0; i < VECTOR_DIMS; i += 4)
                e[i / 4] = (unsigned char)(((int)x[i] & 3) | ((int)x[i + 1] & 3) << 2
                                           | ((int)x[i + 2] & 3) << 4 | ((int)x[i + 3] & 3) << 6);
            break;
        }
        default:    { int32_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (int32_t)x[i]; break; }
    }
}

static void widenCrumbs(const unsigned char *p, int count, int16_t *dst)
{
    int i = 0;
#if defined(__SSE4_1__)
    for (; i + 16 <= count; i += 16) {
        uint32_t w;
        memcpy(&w, p + i / 4, 4); // 4 bytes hold 16 entries
        // Element k of word j holds crumb j of byte k; gather them back into entry order
        __m128i v = _mm_setr_epi32((int)w, (int)(w >> 2), (int)(w >> 4), (int)(w >> 6));
        v = _mm_shuffle_epi8(v, _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15));
        v = _mm_and_si128(v, _mm_set1_epi8(3));
        v = _mm_sub_epi8(_mm_xor_si128(v, _mm_set1_epi8(2)), _mm_set1_epi8(2)); // sign extend 2 bits
        _mm_storeu_si128((__m128i *)(dst + i),     _mm_cvtepi8_epi16(v));
        _mm_storeu_si128((__m128i *)(dst + i + 8), _mm_cvtepi8_epi16(_mm_srli_si128(v, 8)));
    }
#endif
    for (; i < count; i++) {
        int v = (p[i / 4] >> (i % 4) * 2) & 3;
        dst[i] = (int16_t)(v > 1 ? v - 4 : v);
    }
}

static void widenNibbles(const unsigned char *p, int count, int16_t *dst)
{
    int i = 0;
#if defined(__SSE4_1__)
    for (; i + 16 <= count; i += 16) {
        __m128i b  = _mm_loadl_epi64((const __m128i *)(p + i / 2)); // 8 bytes hold 16 entries
        __m128i lo = _mm_and_si128(b, _mm_set1_epi8(0x0F));
        __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(0x0F));
        __m128i v  = _mm_unpacklo_epi8(lo, hi);                     // back into entry order
        v = _mm_sub_epi8(_mm_xor_si128(v, _mm_set1_epi8(8)), _mm_set1_epi8(8)); // sign extend 4 bits
        _mm_storeu_si128((__m128i *)(dst + i),     _mm_cvtepi8_epi16(v));
        _mm_storeu_si128((__m128i *)(dst + i + 8), _mm_cvtepi8_epi16(_mm_srli_si128(v, 8)));
    }
#endif
    for (; i < count; i++) {
        int v = (i & 1) ? p[i / 2] >> 4 : p[i / 2] & 0xF;
        dst[i] = (int16_t)(v > 7 ? v - 16 : v);
    }
}

static inline __attribute__((always_inline)) void widenRun(Real *dst, const void *src, size_t si, DataType from,
                                                           int count)
{
    int i = 0;
    if (si % (size_t)DataPerByte(from) == 0)
        for (; i + VECTOR_DIMS <= count; i += VECTOR_DIMS)
            VectorStore(dst + i, load((const unsigned char *)src + DataSize(from, si + (size_t)i), from));
    for (; i < count; i++) dst[i] = DataGet(src, si + (size_t)i, from);
}

// Copies count entries of type d between any offsets
static void copyEntries(void *dst, size_t di, const void *src, size_t si, DataType d, int count)
{
    size_t per = (size_t)DataPerByte(d);
    if (per > 1 && (di % per || si % per || (size_t)count % per))
        for (int i = 0; i < count; i++) DataSet(dst, di + (size_t)i, d, DataGet(src, si + (size_t)i, d));
    else
        memcpy((unsigned char *)dst + DataSize(d, di), (const unsigned char *)src + DataSize(d, si), DataSize(d, (size_t)count));
}

// Stores n Reals from index, rounding stochastically by key, or to nearest when key is NULL
static inline __attribute__((always_inline)) void quantizeRun(void *base, size_t index, DataType d, const Real *x,
                                                              int n, const uint32_t *key)
{
    for (int k = 0; k < n; k += VECTOR_DIMS) {
        int cnt = n - k < VECTOR_DIMS ? n - k : VECTOR_DIMS;
        size_t e = index + (size_t)k;
        Vector y = VectorZero();
        if (cnt == VECTOR_DIMS) memcpy(&y, x + k, sizeof(y));
        else memcpy(&y, x + k, (size_t)cnt * sizeof(Real));
        unsigned char buffer[VECTOR_DIMS * sizeof(double)];
        int direct = cnt == VECTOR_DIMS && e % (size_t)DataPerByte(d) == 0;
        void *p = direct ? (unsigned char *)base + DataSize(d, e) : buffer;
        if (key) { Vector u = uniformAt(*key, e); storeRounded(p, d, y, &u); }
        else storeRounded(p, d, y, NULL);
        if (!direct) copyEntries(base, e, buffer, 0, d, cnt);
    }
}

static void quantize(void *base, size_t index, DataType d, const Real *x, int n, const uint32_t *key)
{
    switch (d) {
        case INT8:    quantizeRun(base, index, INT8, x, n, key); break;
        case UINT8:   quantizeRun(base, index, UINT8, x, n, key); break;
        case INT16:   quantizeRun(base, index, INT16, x, n, key); break;
        case FLOAT16: quantizeRun(base, index, FLOAT16, x, n, key); break;
        case INT4:    quantizeRun(base, index, INT4, x, n, key); break;
        case INT2:    quantizeRun(base, index, INT2, x, n, key); break;
        default:      quantizeRun(base, index, d, x, n, key); break;
    }
}

void DataConvert(void *dst, size_t di, DataType to, const void *src, size_t si, DataType from, int count)
{
    if (to == FLOAT32 && from != FLOAT32 && sizeof(Real) == 4) { // the stored types worth specialising
        Real *out = (Real *)dst + di;
        switch (from) {
            case INT8:    widenRun(out, src, si, INT8, count); break;
            case UINT8:   widenRun(out, src, si, UINT8, count); break;
            case INT16:   widenRun(out, src, si, INT16, count); break;
            case FLOAT16: widenRun(out, src, si, FLOAT16, count); break;
            case INT4:    widenRun(out, src, si, INT4, count); break;
            case INT2:    widenRun(out, src, si, INT2, count); break;
            default:      widenRun(out, src, si, from, count); break;
        }
        return;
    }

    if (from == FLOAT32 && to != FLOAT32 && sizeof(Real) == 4) {
        quantize(dst, di, to, (const Real *)src + si, count, NULL);
        return;
    }

    if (to == from) { copyEntries(dst, di, src, si, to, count); return; }

    if (to == INT16) {
        int16_t *out = (int16_t *)dst + di;
        switch (from) {
            case INT4:
                if (!(si & 1)) { widenNibbles((const unsigned char *)src + si / 2, count, out); return; }
                break;
            case INT2:
                if (!(si & 3)) { widenCrumbs((const unsigned char *)src + si / 4, count, out); return; }
                break;
            case INT8:  { const int8_t  *e = (const int8_t  *)src + si; for (int i = 0; i < count; i++) out[i] = e[i]; return; }
            case UINT8: { const uint8_t *e = (const uint8_t *)src + si; for (int i = 0; i < count; i++) out[i] = e[i]; return; }
            default: break;
        }
    }

    for (int i = 0; i < count; i++)
        DataSet(dst, di + (size_t)i, to, DataGet(src, si + (size_t)i, from));
}

void DataConvertStochastic(void *dst, size_t di, DataType to, const void *src, size_t si, DataType from, int count,
                           uint32_t key)
{
    if (from == FLOAT32 && sizeof(Real) == 4) { quantize(dst, di, to, (const Real *)src + si, count, &key); return; }

    // Any other type as Reals first
    Real x[VECTOR_DIMS * VECTOR_CARD];
    int chunk = VECTOR_DIMS * VECTOR_CARD;
    for (int c = 0; c < count; c += chunk) {
        int n = count - c < chunk ? count - c : chunk;
        for (int i = 0; i < n; i++) x[i] = DataGet(src, si + (size_t)(c + i), from);
        quantize(dst, di + (size_t)c, to, x, n, &key);
    }
}

Vector VectorLoadAt(const void *base, size_t index, DataType d)
{
    if (index % (size_t)DataPerByte(d) == 0) return VectorLoad((const unsigned char *)base + DataSize(d, index), d);
    Vector x = VectorZero();
    for (int i = 0; i < VECTOR_DIMS; i++) x[i] = DataGet(base, index + (size_t)i, d);
    return x;
}

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
    return VectorZero() - __builtin_convertvector(v > VectorZero(), Vector);
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

Vector VectorFloor(Vector x)
{
    for (int i = 0; i < VECTOR_DIMS; i++)
        x[i] = sizeof(Real) == 8 ? __builtin_floor(x[i]) : __builtin_floorf(x[i]);
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

// MR x NRV block of vector registers, matching the register block in Matrix.c
#define NRV 2
#define NR  (NRV * VECTOR_DIMS)
#define MR  ((VECTOR_CARD - NRV - 1) / NRV)

void VectorDot(void *A, void *B, Real *C, Real *q, int n, int k)
{
    int32_t acc[MR][NRV] __attribute__((__vector_size__(VECTOR_DIMS * 4))) = {0};
    int kPairs = (k + 1) / 2;

    const Real *scale = A;
    const int16_t *aRow = (const int16_t *)(scale + MR);
    const int16_t *bRow = B;
    for (int p2 = 0; p2 < kPairs; p2++, aRow += 2 * MR, bRow += 2 * NR) {
        int32_t b[NRV] __attribute__((__vector_size__(VECTOR_DIMS * 4)));
        for (int v = 0; v < NRV; v++)
            memcpy(&b[v], bRow + v * 2 * VECTOR_DIMS, sizeof(b[v]));

        for (int mr = 0; mr < MR; mr++)
            for (int v = 0; v < NRV; v++) {
                int32_t c __attribute__((__vector_size__(VECTOR_DIMS * 4))) = acc[mr][v];
#if defined(__AVX512BW__)
                if (sizeof(Real) == 4) {
                    int32_t pair;
                    memcpy(&pair, aRow + 2 * mr, sizeof(pair));
                    __m512i bv, cv;
                    memcpy(&bv, &b[v], sizeof(b[v]));
                    memcpy(&cv, &c, sizeof(c));
                    cv = _mm512_add_epi32(cv, _mm512_madd_epi16(_mm512_set1_epi32(pair), bv));
                    memcpy(&c, &cv, sizeof(c));
                    acc[mr][v] = c;
                    continue;
                }
#elif defined(__AVX2__) && !defined(__AVX512F__)
                if (sizeof(Real) == 4) {
                    int32_t pair;
                    memcpy(&pair, aRow + 2 * mr, sizeof(pair));
                    __m256i bv, cv;
                    memcpy(&bv, &b[v], sizeof(b[v]));
                    memcpy(&cv, &c, sizeof(c));
                    cv = _mm256_add_epi32(cv, _mm256_madd_epi16(_mm256_set1_epi32(pair), bv));
                    memcpy(&c, &cv, sizeof(c));
                    acc[mr][v] = c;
                    continue;
                }
#endif
                for (int i = 0; i < VECTOR_DIMS; i++)
                    c[i] += (int32_t)aRow[2 * mr] * bRow[v * 2 * VECTOR_DIMS + 2 * i]
                          + (int32_t)aRow[2 * mr + 1] * bRow[v * 2 * VECTOR_DIMS + 2 * i + 1];
                acc[mr][v] = c;
            }
    }

    for (int mr = 0; mr < MR; mr++) {
        Real *Crow = C + (size_t)mr * n;
        Vector s = VectorBcast(scale[mr]);
        for (int v = 0; v < NRV; v++) {
            Vector sum = VectorZero();
            for (int i = 0; i < VECTOR_DIMS; i++)
                sum[i] = (Real)acc[mr][v][i];
            if (q)
                sum = VectorMul(sum, VectorLoad(q + v * VECTOR_DIMS, FLOAT32));
            VectorStore(Crow + v * VECTOR_DIMS, VectorFma(s, sum, VectorLoad(Crow + v * VECTOR_DIMS, FLOAT32)));
        }
    }
}

// exp(a) with ~1e-6 (float) or ~1e-13 (double) relative error
Vector expe(Vector a)
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
Vector loge(Vector a)
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
Vector relu(Vector a)     { return VectorMax(a, VectorZero()); }
Vector drelu(Vector a)    { return VectorGt0(a); }
Vector identity(Vector a) { return a; }
Vector one(Vector a)      { (void)a; return VectorBcast(1.0); }
Vector negate(Vector a)   { return VectorSub(VectorZero(), a); }
Vector minusOne(Vector a) { (void)a; return VectorBcast(-1.0); }
Vector sigmoid(Vector a)  { return VectorDiv(VectorBcast(1.0), VectorAdd(VectorBcast(1.0), expe(VectorSub(VectorZero(), a)))); }
Vector dsigmoid(Vector a) { Vector s = sigmoid(a); return VectorMul(s, VectorSub(VectorBcast(1.0), s)); }
Vector tanhe(Vector a)    { return VectorSub(VectorDiv(VectorBcast(2.0), VectorAdd(VectorBcast(1.0), expe(VectorMul(VectorBcast(-2.0), a)))), VectorBcast(1.0)); }
Vector dtanhe(Vector a)   { Vector t = tanhe(a); return VectorSub(VectorBcast(1.0), VectorMul(t, t)); }
Vector dloge(Vector a)    { return VectorDiv(VectorBcast(1.0), a); }
Vector sqrte(Vector a)    { return VectorSqrt(a); }
Vector dsqrte(Vector a)   { return VectorDiv(VectorBcast(0.5), VectorSqrt(a)); }
static Vector geluArg(Vector a) { return VectorMul(VectorBcast(0.7978845608028654), VectorAdd(a, VectorMul(VectorBcast(0.044715), VectorMul(a, VectorMul(a, a))))); } // 0.797... = sqrt(2/pi)
Vector gelu(Vector a)     { return VectorMul(VectorBcast(0.5), VectorMul(a, VectorAdd(VectorBcast(1.0), tanhe(geluArg(a))))); }
Vector dgelu(Vector a)
{
    Vector c = VectorBcast(0.7978845608028654); // sqrt(2/pi)
    Vector u = tanhe(geluArg(a));
    Vector t1 = VectorMul(VectorBcast(0.5), VectorAdd(VectorBcast(1.0), u));
    Vector t2 = VectorMul(VectorMul(VectorMul(VectorBcast(0.5), a), VectorSub(VectorBcast(1.0), VectorMul(u, u))),
                VectorMul(c, VectorAdd(VectorBcast(1.0), VectorMul(VectorBcast(0.134145), VectorMul(a, a)))));
    return VectorAdd(t1, t2);
}

// Binary functions and partial derivatives for vectors
Vector add(Vector a, Vector b)       { return VectorAdd(a, b); }
Vector subtract(Vector a, Vector b)  { return VectorSub(a, b); }
Vector multiply(Vector a, Vector b)  { return VectorMul(a, b); }
Vector divide(Vector a, Vector b)    { return VectorDiv(a, b); }
Vector max(Vector a, Vector b)       { return VectorMax(a, b); }
Vector first(Vector a, Vector b)     { (void)b; return a; }
Vector second(Vector a, Vector b)    { (void)a; return b; }
Vector ones(Vector a, Vector b)      { (void)a; (void)b; return VectorBcast(1.0); }
Vector minusOnes(Vector a, Vector b) { (void)a; (void)b; return VectorBcast(-1.0); }
Vector divideDa(Vector a, Vector b)  { (void)a; return VectorDiv(VectorBcast(1.0), b); }
Vector divideDb(Vector a, Vector b)  { return VectorSub(VectorZero(), VectorDiv(a, VectorMul(b, b))); }
Vector maxDa(Vector a, Vector b)     { return VectorSub(VectorBcast(1.0), VectorGt0(VectorSub(b, a))); }
Vector maxDb(Vector a, Vector b)     { return VectorGt0(VectorSub(b, a)); }
Vector min(Vector a, Vector b)       { return VectorMin(a, b); }
Vector minDa(Vector a, Vector b)     { return VectorSub(VectorBcast(1.0), VectorGt0(VectorSub(a, b))); }
Vector minDb(Vector a, Vector b)     { return VectorGt0(VectorSub(a, b)); }

Real VectorFirst(Vector v)                        { return v[0]; }
Real VectorApply(VectorBinary fn, Real a, Real b) { return VectorFirst(fn(VectorBcast(a), VectorBcast(b))); }
