#ifndef VECTOR_H
#define VECTOR_H

#include "../tensorgrid.h" // Real

/*
 * The Vector type is a SIMD register of Reals, sized at compile time from
 * the available SIMD instruction set. VECTOR_DIMS is how many Reals fit in
 * a Vector and VECTOR_CARD is how many vector registers are available.
 */
#if defined(__AVX512F__)
#define VECTOR_BYTES 64
#define VECTOR_CARD 32
#elif defined(__AVX2__)
#define VECTOR_BYTES 32
#define VECTOR_CARD 16
#else
#define VECTOR_BYTES sizeof(Real)
#define VECTOR_CARD 16
#endif
#define VECTOR_DIMS ((int)(VECTOR_BYTES / sizeof(Real)))

typedef Real Vector __attribute__((__vector_size__(VECTOR_BYTES)));
typedef Vector (*VectorUnary)(Vector);
typedef Vector (*VectorBinary)(Vector, Vector);

Vector VectorLoad(const void *p, DataType d);
void   VectorStore(Real *p, Vector x);
void   VectorFetch(const Real *p);
Vector VectorBcast(Real x);
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
Real   VectorHsum(Vector v);

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
