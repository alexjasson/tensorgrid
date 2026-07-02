#ifndef VECTOR_H
#define VECTOR_H

#include <immintrin.h>

/*
 * The Vector type and it's dimension is determined at compile time from the
 * available SIMD instruction set. VECTOR_DIMS is how many floats fit in a
 * Vector and VECTOR_CARD is how many vector registers are available.
 */
#if defined(__AVX512F__)
typedef __m512 Vector;
#define VECTOR_DIMS 16
#define VECTOR_CARD 32
#elif defined(__AVX2__)
typedef __m256 Vector;
#define VECTOR_DIMS 8
#define VECTOR_CARD 16
#else
typedef float Vector;
#define VECTOR_DIMS 1
#define VECTOR_CARD 16
#endif

typedef Vector (*VectorUnary)(Vector);
typedef Vector (*VectorBinary)(Vector, Vector);

Vector VectorLoad(const float *p);
void   VectorStore(float *p, Vector x);
void   VectorFetch(const float *p);
Vector VectorBcast(float x);
Vector VectorAdd(Vector a, Vector b);
Vector VectorSub(Vector a, Vector b);
Vector VectorMul(Vector a, Vector b);
Vector VectorDiv(Vector a, Vector b);
Vector VectorFma(Vector a, Vector b, Vector c);
Vector VectorZero(void);
Vector VectorMax(Vector a, Vector b);
Vector VectorMin(Vector a, Vector b);
Vector VectorFnmadd(Vector a, Vector b, Vector c);
Vector VectorRound(Vector x);
Vector VectorPow2i(Vector n);
Vector VectorSqrt(Vector x);
Vector VectorLog2i(Vector x);
Vector VectorMantissa(Vector x);
Vector VectorGt0(Vector v);
float  VectorHsum(Vector v);

// Unary activation functions and their derivatives
Vector reluv(Vector a);
Vector dreluv(Vector a);
Vector identityv(Vector a);
Vector didentityv(Vector a);
Vector negatev(Vector a);
Vector dnegatev(Vector a);
Vector sigmoidv(Vector a);
Vector dsigmoidv(Vector a);
Vector tanhv(Vector a);
Vector dtanhv(Vector a);
Vector expv(Vector a);
Vector logv(Vector a);
Vector dlogv(Vector a);
Vector sqrtv(Vector a);
Vector dsqrtv(Vector a);
Vector geluv(Vector a);
Vector dgeluv(Vector a);

// Binary functions and their partial derivatives
Vector addv(Vector a, Vector b);
Vector subtractv(Vector a, Vector b);
Vector multiplyv(Vector a, Vector b);
Vector dividev(Vector a, Vector b);
Vector maxv(Vector a, Vector b);
Vector firstv(Vector a, Vector b);
Vector secondv(Vector a, Vector b);
Vector onesv(Vector a, Vector b);
Vector minusOnesv(Vector a, Vector b);
Vector divideDav(Vector a, Vector b);
Vector divideDbv(Vector a, Vector b);
Vector maxDav(Vector a, Vector b);
Vector maxDbv(Vector a, Vector b);
Vector minv(Vector a, Vector b);
Vector minDav(Vector a, Vector b);
Vector minDbv(Vector a, Vector b);

#endif
