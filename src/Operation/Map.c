// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Vector.h"
#include "../math/Scalar.h"

typedef struct { Unary u; } MapContext;

static void OperationMapForward(Operation *op);
static void OperationMapBackward(Operation *op);
static void mapForward(void *arg, int start, int end);
static void mapBackward(void *arg, int start, int end);

Tensor OperationMap(Unary fn, Tensor A)
{
    MapContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->u = fn;
    Tensor out = TensorNew(TensorDimensions(A), TensorShape(A), REAL, TensorGradients(A) != NULL);
    return OperationExecute(OperationNew(out, A, NULL, OperationMapForward, OperationMapBackward, ctx));
}

static void OperationMapForward(Operation *op)
{
    OperationDispatch(mapForward, op, TensorSize(op->out), TensorSize(op->out));
}

static void OperationMapBackward(Operation *op)
{
    if (!TensorGradients(op->A)) return;
    OperationDispatch(mapBackward, op, TensorSize(op->out), TensorSize(op->out));
}


static void mapForward(void *arg, int start, int end)
{
    Operation  *op  = arg;
    MapContext *ctx = op->context;
    Real      *out = TensorEntries(op->out);
    Real      *A   = TensorEntries(op->A);
    VectorUnary fn = (VectorUnary)ctx->u.fv;

    if (TensorDataType(op->A) != REAL || TensorScale(op->A)) {
        ScalarUnary sfn = (ScalarUnary)ctx->u.f;
        int aC = isContiguous(TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A));
        for (int i = start; i < end; i++) {
            int off = aC ? i : unravel(i, TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A));
            out[i] = sfn(TensorGet(op->A, off));
        }
        return;
    }

    if (isContiguous(TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A))) {
        int i = start;
        for (; i + VECTOR_DIMS <= end; i += VECTOR_DIMS)
            VectorStore(out + i, fn(VectorLoad(A + i, REAL)));
        if (i < end) {
            Real x[VECTOR_DIMS], y[VECTOR_DIMS];
            int tail = end - i;
            for (int t = 0; t < VECTOR_DIMS; t++)
                x[t] = t < tail ? A[i + t] : 1.0f;
            VectorStore(y, fn(VectorLoad(x, REAL)));
            for (int t = 0; t < tail; t++)
                out[i + t] = y[t];
        }
    } else {
        for (int i = start; i < end; i += VECTOR_DIMS) {
            Real x[VECTOR_DIMS], y[VECTOR_DIMS];
            int tail = end - i < VECTOR_DIMS ? end - i : VECTOR_DIMS;
            for (int t = 0; t < VECTOR_DIMS; t++) {
                int j = i + t;
                x[t] = t < tail ? A[unravel(j, TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A))] : 1.0f;
            }
            VectorStore(y, fn(VectorLoad(x, REAL)));
            for (int t = 0; t < tail; t++)
                out[i + t] = y[t];
        }
    }
}

static void mapBackward(void *arg, int start, int end)
{
    Operation   *op = arg;
    MapContext *ctx = op->context;
    Real       *A   = TensorEntries(op->A);
    Real       *dA  = TensorGradients(op->A);
    Real     *dout  = TensorGradients(op->out);
    VectorUnary dafn = (VectorUnary)ctx->u.dav;
    if (isContiguous(TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A))) {
        int j = start;
        for (; j + VECTOR_DIMS <= end; j += VECTOR_DIMS)
            VectorStore(dA + j, VectorFma(VectorLoad(dout + j, REAL), dafn(VectorLoad(A + j, REAL)), VectorLoad(dA + j, REAL)));
        if (j < end) {
            Real x[VECTOR_DIMS], g[VECTOR_DIMS], old[VECTOR_DIMS], y[VECTOR_DIMS];
            int tail = end - j;
            for (int t = 0; t < VECTOR_DIMS; t++) {
                x[t] = t < tail ? A[j + t] : 1.0f;
                g[t] = t < tail ? dout[j + t] : 0.0f;
                old[t] = t < tail ? dA[j + t] : 0.0f;
            }
            VectorStore(y, VectorFma(VectorLoad(g, REAL), dafn(VectorLoad(x, REAL)), VectorLoad(old, REAL)));
            for (int t = 0; t < tail; t++)
                dA[j + t] = y[t];
        }
    } else {
        for (int j = start; j < end; j += VECTOR_DIMS) {
            Real x[VECTOR_DIMS], g[VECTOR_DIMS], y[VECTOR_DIMS];
            int offs[VECTOR_DIMS];
            int tail = end - j < VECTOR_DIMS ? end - j : VECTOR_DIMS;
            for (int t = 0; t < VECTOR_DIMS; t++) {
                int i = j + t;
                offs[t] = 0;
                x[t] = 1.0f;
                g[t] = 0.0f;
                if (t < tail) {
                    offs[t] = unravel(i, TensorShape(op->A), TensorStride(op->A), TensorDimensions(op->A));
                    x[t] = A[offs[t]];
                    g[t] = dout[i];
                }
            }
            VectorStore(y, VectorMul(VectorLoad(g, REAL), dafn(VectorLoad(x, REAL))));
            for (int t = 0; t < tail; t++)
                dA[offs[t]] += y[t];
        }
    }
}
