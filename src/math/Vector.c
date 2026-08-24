#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include "Vector.h"

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

// Size of the data type in bytes
size_t DataSize(DataType d) { return (0x8424211u >> (4 * d)) & 0xF; }

int64_t DataMax(DataType d)
{
    switch (d) {
        case UINT8: return UINT8_MAX;
        case INT8:  return INT8_MAX;
        case INT16: return INT16_MAX;
        case INT32: return INT32_MAX;
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
        default:    return 0;
    }
}

static Real randomLane(int lane)
{
    static uint64_t state[VECTOR_DIMS];
    if (!state[0])
        for (int i = 0; i < VECTOR_DIMS; i++)
            state[i] = 0x853c49e6748fea9bULL + 0x9e3779b97f4a7c15ULL * (uint64_t)(i + 1);

    uint64_t s = state[lane];
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    state[lane] = s;
    return (Real)(s >> 40) * (Real)(1.0 / 16777216.0);
}

static Vector randomVector(void)
{
    Vector u;
    for (int i = 0; i < VECTOR_DIMS; i++) u[i] = randomLane(i);
    return u;
}

static Real roundDown(Real x)
{
    int64_t i = (int64_t)x;
    return (Real)(i - (x < (Real)i));
}

static void DataSetInt64(void *p, DataType d, int64_t x)
{
    if (d > INT32) { DataSet(p, d, (Real)x); return; }
    int64_t low = DataMin(d), high = DataMax(d);
    if (x < low)  x = low;
    if (x > high) x = high;
    switch (d) {
        case UINT8: *(unsigned char *)p = (unsigned char)x; break;
        case INT8:  *(signed char *)p = (signed char)x; break;
        case INT16: { int16_t v = (int16_t)x; memcpy(p, &v, 2); break; }
        default:    { int32_t v = (int32_t)x; memcpy(p, &v, 4); break; }
    }
}

void DataSetStochastic(void *p, DataType d, Real x)
{
    if (!DataMax(d)) { DataSet(p, d, x); return; }
    DataSetInt64(p, d, (int64_t)roundDown(x + randomLane(0)));
}

Real DataGet(const void *p, DataType d)
{
    switch (d) {
        case UINT8:   return (Real)*(const unsigned char *)p;
        case INT8:    return (Real)*(const signed char *)p;
        case INT16:   { int16_t  x; memcpy(&x, p, 2); return (Real)x; }
        case INT32:   { int32_t  x; memcpy(&x, p, 4); return (Real)x; }
        case FLOAT16: { uint16_t x; memcpy(&x, p, 2); return halfToReal(x); }
        case FLOAT32: { float    x; memcpy(&x, p, 4); return (Real)x; }
        case FLOAT64: { double   x; memcpy(&x, p, 8); return (Real)x; }
    }
    return 0.0;
}

void DataSet(void *p, DataType d, Real x)
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
    }
}

Vector VectorLoad(const void *p, DataType d)
{
    Vector x = VectorZero();
    switch (d) {
        case UINT8:   { const uint8_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
        case INT8:    { const int8_t  *e = p; for (int i = 0; i < VECTOR_DIMS; i++) x[i] = (Real)e[i]; break; }
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
                x[i] = DataGet((const uint16_t *)p + i, FLOAT16);
            break;
    }
    return x;
}
void   VectorStore(Real *p, Vector x)             { memcpy(p, &x, sizeof(x)); }

void VectorStoreStochastic(void *p, DataType d, Vector x)
{
    if (!DataMax(d)) {
        if (d == FLOAT32 && sizeof(Real) == 4) { memcpy(p, &x, sizeof(x)); return; }
        for (int i = 0; i < VECTOR_DIMS; i++)
            DataSet((unsigned char *)p + (size_t)i * DataSize(d), d, x[i]);
        return;
    }

    x = VectorMin(VectorMax(x, VectorBcast((Real)DataMin(d))), VectorBcast((Real)DataMax(d)));
    x = x + randomVector();
    for (int i = 0; i < VECTOR_DIMS; i++) x[i] = roundDown(x[i]);
    switch (d) {
        case UINT8: { uint8_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (uint8_t)x[i]; break; }
        case INT8:  { int8_t  *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (int8_t)x[i];  break; }
        case INT16: { int16_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (int16_t)x[i]; break; }
        default:    { int32_t *e = p; for (int i = 0; i < VECTOR_DIMS; i++) e[i] = (int32_t)x[i]; break; }
    }
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
