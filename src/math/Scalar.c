#include <stdint.h>
#include "Scalar.h"

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
