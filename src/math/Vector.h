#ifndef VECTOR_H
#define VECTOR_H

#include <stddef.h>
#include <stdint.h>
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

size_t  DataSize(DataType d, size_t n);    // Bytes needed to store n entries
int     DataPerByte(DataType d);           // Entries packed into one byte: 2 for INT4, 4 for INT2, else 1
int64_t DataMax(DataType d);
int64_t DataMin(DataType d);
Real    DataGet(const void *base, size_t index, DataType d);
void    DataSet(void *base, size_t index, DataType d, Real x);
void    DataConvert(void *dst, size_t di, DataType to, const void *src, size_t si, DataType from, int count);
void    DataConvertStochastic(void *dst, size_t di, DataType to, const void *src, size_t si, DataType from, int count,
                              uint32_t key); // DataConvert, rounding integers stochastically by (key, di)

Vector VectorLoad(const void *p, DataType d);
void   VectorStore(Real *p, Vector x);
Vector VectorLoadPartial(const Real *p, int n);         // n <= VECTOR_DIMS Reals from p, the rest of the Vector zero
void   VectorStorePartial(Real *p, Vector x, int n);    // The first n <= VECTOR_DIMS Reals of x to p
Vector VectorLoadAt(const void *base, size_t index, DataType d);
Vector VectorUniformAt(uint32_t key, size_t index);     // The [0, 1) dither DataConvertStochastic uses for (key, index)
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
Vector VectorFloor(Vector x);
Vector VectorPow2i(Vector n);
Vector VectorSqrt(Vector x);
Vector VectorLog2i(Vector x);
Vector VectorMantissa(Vector x);
Vector VectorGt0(Vector v);
Real   VectorHsum(Vector v);
void VectorDot(void *A, void *B, Real *C, Real *q, int n, int k); // C[i * n + j] += q[j] * A[i] . B[j] for A[i], B[j] in R^k
Real VectorFirst(Vector v);
Real VectorApply(VectorBinary fn, Real a, Real b);

// Unary activation functions and their derivatives
Vector relu(Vector a);
Vector drelu(Vector a);
Vector identity(Vector a);
Vector one(Vector a);
Vector negate(Vector a);
Vector minusOne(Vector a);
Vector sigmoid(Vector a);
Vector dsigmoid(Vector a);
Vector tanhe(Vector a);
Vector dtanhe(Vector a);
Vector expe(Vector a);
Vector loge(Vector a);
Vector dloge(Vector a);
Vector sqrte(Vector a);
Vector dsqrte(Vector a);
Vector gelu(Vector a);
Vector dgelu(Vector a);

// Binary functions and their partial derivatives
Vector add(Vector a, Vector b);
Vector subtract(Vector a, Vector b);
Vector multiply(Vector a, Vector b);
Vector divide(Vector a, Vector b);
Vector max(Vector a, Vector b);
Vector first(Vector a, Vector b);
Vector second(Vector a, Vector b);
Vector ones(Vector a, Vector b);
Vector minusOnes(Vector a, Vector b);
Vector divideDa(Vector a, Vector b);
Vector divideDb(Vector a, Vector b);
Vector maxDa(Vector a, Vector b);
Vector maxDb(Vector a, Vector b);
Vector min(Vector a, Vector b);
Vector minDa(Vector a, Vector b);
Vector minDb(Vector a, Vector b);

#endif
