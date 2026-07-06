#ifndef SCALAR_H
#define SCALAR_H

#include "../tensorgrid.h" // Real

typedef Real Scalar;

typedef Scalar (*ScalarUnary)(Scalar);
typedef Scalar (*ScalarBinary)(Scalar, Scalar);

// Unary functions and derivatives
Scalar relu(Scalar a);
Scalar drelu(Scalar a);
Scalar sigmoid(Scalar a);
Scalar dsigmoid(Scalar a);
Scalar loge(Scalar a);
Scalar dloge(Scalar a);
Scalar negate(Scalar a);
Scalar identity(Scalar a);
Scalar one(Scalar a);
Scalar minusOne(Scalar a);
Scalar expe(Scalar a);
Scalar tanhe(Scalar a);
Scalar dtanhe(Scalar a);
Scalar sqrte(Scalar a);
Scalar dsqrte(Scalar a);
Scalar gelu(Scalar a);
Scalar dgelu(Scalar a);

// Binary functions and partial derivatives
Scalar add(Scalar a, Scalar b);
Scalar subtract(Scalar a, Scalar b);
Scalar multiply(Scalar a, Scalar b);
Scalar divide(Scalar a, Scalar b);
Scalar max(Scalar a, Scalar b);
Scalar first(Scalar a, Scalar b);
Scalar second(Scalar a, Scalar b);
Scalar ones(Scalar a, Scalar b);
Scalar minusOnes(Scalar a, Scalar b);
Scalar divideDa(Scalar a, Scalar b);
Scalar divideDb(Scalar a, Scalar b);
Scalar maxDa(Scalar a, Scalar b);
Scalar maxDb(Scalar a, Scalar b);
Scalar min(Scalar a, Scalar b);
Scalar minDa(Scalar a, Scalar b);
Scalar minDb(Scalar a, Scalar b);

#endif
