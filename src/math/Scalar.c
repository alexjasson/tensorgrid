#include <stdint.h>
#include <string.h>
#include "Scalar.h"

// Restrict x to the range [lo, hi]
static Real in(Real x, Real lo, Real hi) { return x < lo ? lo : x > hi ? hi : x; }
// Integer truncates to nearest halves away from zero
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
size_t ScalarSize(DataType d) { return ((0x8424211u | (unsigned)sizeof(Real) << 28) >> (4 * d)) & 0xF; }

Real ScalarLoad(const void *p, DataType d)
{
    switch (d) {
        case UINT8:   return (Real)*(const unsigned char *)p;
        case INT8:    return (Real)*(const signed char *)p;
        case INT16:   { int16_t  x; memcpy(&x, p, 2); return (Real)x; }
        case INT32:   { int32_t  x; memcpy(&x, p, 4); return (Real)x; }
        case FLOAT16: { uint16_t x; memcpy(&x, p, 2); return halfToReal(x); }
        case FLOAT32: { float    x; memcpy(&x, p, 4); return (Real)x; }
        case FLOAT64: { double   x; memcpy(&x, p, 8); return (Real)x; }
        case REAL:    { Real     x; memcpy(&x, p, sizeof(Real)); return x; }
    }
    return 0.0;
}

void ScalarStore(void *p, DataType d, Real x)
{
    switch (d) {
        case UINT8:   *(unsigned char *)p = (unsigned char)nearest(in(x, 0.0, 255.0)); break;
        case INT8:    *(signed char *)p = (signed char)nearest(in(x, -128.0, 127.0)); break;
        case INT16:   { int16_t  v = (int16_t)nearest(in(x, -32768.0, 32767.0));           memcpy(p, &v, 2); break; }
        case INT32:   { int32_t  v = (int32_t)nearest(in(x, -2147483648.0, 2147483520.0)); memcpy(p, &v, 4); break; }
        case FLOAT16: { uint16_t v = realToHalf(x);                                        memcpy(p, &v, 2); break; }
        case FLOAT32: { float    v = (float)in(x, -3.402823466e+38, 3.402823466e+38);      memcpy(p, &v, 4); break; }
        case FLOAT64: { double   v = (double)x;                                            memcpy(p, &v, 8); break; }
        case REAL:    memcpy(p, &x, sizeof(Real)); break;
    }
}

// exp(a) with ~1e-5 (float) or ~1e-13 (double) relative error
Scalar expe(Scalar a)
{
    Real expMax = sizeof(Real) == 8 ?  709.0 :  88.0;
    Real expMin = sizeof(Real) == 8 ? -708.0 : -87.3;
    if (a > expMax) a = expMax;
    if (a < expMin) return 0.0;

    int  n = (int)(a * (Real)1.4426950408889634 + (a >= 0.0 ? (Real)0.5 : (Real)-0.5));
    Real r = a - (Real)n * (Real)0.6931471805599453;

    // Taylor series: 1 + r + r^2/2 + r^3/6 + ...
    Real p = (Real)(1.0 / 120.0);
    if (sizeof(Real) == 8) {
        p = 1.0 / 6227020800.0;            // 1/13!
        p = p * r + 1.0 / 479001600.0;     // 1/12!
        p = p * r + 1.0 / 39916800.0;      // 1/11!
        p = p * r + 1.0 / 3628800.0;       // 1/10!
        p = p * r + 1.0 / 362880.0;        // 1/9!
        p = p * r + 1.0 / 40320.0;         // 1/8!
        p = p * r + 1.0 / 5040.0;          // 1/7!
        p = p * r + 1.0 / 720.0;           // 1/6!
        p = p * r + 1.0 / 120.0;
    }
    p = p * r + (Real)(1.0 / 24.0);
    p = p * r + (Real)(1.0 / 6.0);
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;

    // Multiply by 2^n
    if (sizeof(Real) == 8) {
        union { double f; int64_t i; } u = { .i = (int64_t)(1023 + n) << 52 };
        return p * u.f;
    } else {
        union { float f; int32_t i; } u = { .i = (127 + n) << 23 };
        return p * u.f;
    }
}

// log(a) with ~1e-5 (float) or ~1e-11 (double) relative error
Scalar loge(Scalar a)
{
    if (a <= 0.0) return -INFINITE;

    int  e;
    Real m;
    if (sizeof(Real) == 8) {
        union { double f; int64_t i; } u = { .f = a };
        e   = (int)((u.i >> 52) & 0x7FF) - 1023;
        u.i = (u.i & 0x000FFFFFFFFFFFFFLL) | 0x3FF0000000000000LL;
        m   = u.f;
    } else {
        union { float f; int32_t i; } u = { .f = a };
        e   = ((u.i >> 23) & 0xFF) - 127;
        u.i = (u.i & 0x007FFFFF) | 0x3F800000;
        m   = u.f;
    }

    Real t  = (m - (Real)1.0) / (m + (Real)1.0);
    Real t2 = t * t;
    Real p = (Real)(1.0 / 7.0);
    if (sizeof(Real) == 8) {
        p = 1.0 / 19.0;
        p = p * t2 + 1.0 / 17.0;
        p = p * t2 + 1.0 / 15.0;
        p = p * t2 + 1.0 / 13.0;
        p = p * t2 + 1.0 / 11.0;
        p = p * t2 + 1.0 / 9.0;
        p = p * t2 + 1.0 / 7.0;
    }
    p = p * t2 + (Real)(1.0 / 5.0);
    p = p * t2 + (Real)(1.0 / 3.0);
    p = p * t2 + (Real)1.0;
    Real l = (Real)2.0 * t * p;

    return l + (Real)e * (Real)0.6931471805599453;
}

// Unary functions and derivatives
Scalar relu(Scalar a)     { return a > 0.0 ? a : 0.0; }
Scalar drelu(Scalar a)    { return a > 0.0 ? 1.0 : 0.0; }
Scalar sigmoid(Scalar a)  { return (Real)1.0 / ((Real)1.0 + expe(-a)); }
Scalar dsigmoid(Scalar a) { Scalar s = sigmoid(a); return s * ((Real)1.0 - s); }
Scalar dloge(Scalar a)    { return (Real)1.0 / a; }
Scalar negate(Scalar a)   { return -a; }
Scalar identity(Scalar a) { return a; }
Scalar one(Scalar a)      { (void)a; return  1.0; }
Scalar minusOne(Scalar a) { (void)a; return -1.0; }
Scalar tanhe(Scalar a)    { return (Real)2.0 / ((Real)1.0 + expe((Real)-2.0 * a)) - (Real)1.0; }
Scalar dtanhe(Scalar a)   { Scalar t = tanhe(a); return (Real)1.0 - t * t; }
Scalar sqrte(Scalar a)    { return sizeof(Real) == 8 ? __builtin_sqrt(a) : __builtin_sqrtf(a); }
Scalar dsqrte(Scalar a)   { return (Real)0.5 / sqrte(a); }
Scalar gelu(Scalar a)     { return (Real)0.5 * a * ((Real)1.0 + tanhe((Real)0.7978845608028654 * (a + (Real)0.044715 * a * a * a))); } // 0.797... = sqrt(2/pi)
Scalar dgelu(Scalar a)
{
    Scalar c = 0.7978845608028654; // sqrt(2/pi)
    Scalar u = tanhe(c * (a + (Real)0.044715 * a * a * a));
    return (Real)0.5 * ((Real)1.0 + u) + (Real)0.5 * a * ((Real)1.0 - u * u) * c * ((Real)1.0 + (Real)0.134145 * a * a);
}

// Binary functions and partial derivatives
Scalar add(Scalar a, Scalar b)       { return a + b; }
Scalar subtract(Scalar a, Scalar b)  { return a - b; }
Scalar multiply(Scalar a, Scalar b)  { return a * b; }
Scalar divide(Scalar a, Scalar b)    { return a / b; }
Scalar max(Scalar a, Scalar b)       { return a > b ? a : b; }
Scalar first(Scalar a, Scalar b)     { (void)b; return a; }
Scalar second(Scalar a, Scalar b)    { (void)a; return b; }
Scalar ones(Scalar a, Scalar b)      { (void)a; (void)b; return  1.0; }
Scalar minusOnes(Scalar a, Scalar b) { (void)a; (void)b; return -1.0; }
Scalar divideDa(Scalar a, Scalar b)  { (void)a; return (Real)1.0 / b; }
Scalar divideDb(Scalar a, Scalar b)  { return -a / (b * b); }
Scalar maxDa(Scalar a, Scalar b)     { return a >= b ? 1.0 : 0.0; }
Scalar maxDb(Scalar a, Scalar b)     { return b > a ? 1.0 : 0.0; }
Scalar min(Scalar a, Scalar b)       { return a < b ? a : b; }
Scalar minDa(Scalar a, Scalar b)     { return a <= b ? 1.0 : 0.0; }
Scalar minDb(Scalar a, Scalar b)     { return b < a ? 1.0 : 0.0; }
