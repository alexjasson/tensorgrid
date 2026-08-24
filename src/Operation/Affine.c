// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Vector.h"

typedef struct { Unary u; int inner; } AffineContext;

static void OperationAffineForward(Operation *op);
static void OperationAffineBackward(Operation *op);
static void affineForward(void *arg, int start, int end);
static void affineBackwardDa(void *arg, int start, int end);
static void affineBackwardDb(void *arg, int start, int end);

Tensor OperationAffine(Unary fn, Tensor A, Tensor bias, int inner)
{
    AffineContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->u = fn;
    ctx->inner = inner;

    int autograd = TensorGradients(A) != NULL || TensorGradients(bias) != NULL;
    Tensor out = TensorNew(TensorDimensions(A), TensorShape(A), FLOAT32, autograd);
    return OperationExecute(OperationNew(out, A, bias, OperationAffineForward, OperationAffineBackward, ctx));
}

static void OperationAffineForward(Operation *op)
{
    AffineContext *ctx = op->context;
    int outer = TensorSize(op->out) / (TensorSize(op->B) * ctx->inner);
    OperationDispatch(affineForward, op, outer, TensorSize(op->out));
}

static void OperationAffineBackward(Operation *op)
{
    AffineContext *ctx = op->context;
    int biasSize = TensorSize(op->B);
    int outer    = TensorSize(op->out) / (biasSize * ctx->inner);
    if (TensorGradients(op->A))
        OperationDispatch(affineBackwardDa, op, outer, TensorSize(op->out));
    if (TensorGradients(op->B))
        OperationDispatch(affineBackwardDb, op, biasSize, TensorSize(op->out));
}


// out = f(A + bias)
static void affineForward(void *arg, int start, int end)
{
    Operation *op = arg;
    AffineContext *ctx = op->context;
    Real *out = TensorEntries(op->out), *A = TensorEntries(op->A);
    int biasSize = TensorSize(op->B), inner = ctx->inner;
    VectorUnary fn = (VectorUnary)ctx->u.f;

    Real bias[biasSize];
    for (int i = 0; i < biasSize; i++) bias[i] = TensorGet(op->B, i);

    if (TensorDataType(op->A) != FLOAT32 || ScaleGet(op->A).values) {
        for (int o = start; o < end; o++)
            for (int c = 0; c < biasSize; c++) {
                Real bc = bias[c];
                int base = (o * biasSize + c) * inner;
                for (int i = 0; i < inner; i++)
                    out[base + i] = VectorFirst(fn(VectorBcast(TensorGet(op->A, base + i) + bc)));
            }
        return;
    }

    if (inner == 1) {
        for (int o = start; o < end; o++) {
            size_t base = (size_t)o * biasSize;
            int c = 0;
            for (; c + VECTOR_DIMS <= biasSize; c += VECTOR_DIMS)
                VectorStore(out + base + c, fn(VectorAdd(VectorLoad(A + base + c, FLOAT32), VectorLoad(bias + c, FLOAT32))));
            if (c < biasSize) {
                Real x[VECTOR_DIMS], y[VECTOR_DIMS];
                int tail = biasSize - c;
                for (int t = 0; t < VECTOR_DIMS; t++)
                    x[t] = t < tail ? A[base + c + t] + bias[c + t] : 0.0f;
                VectorStore(y, fn(VectorLoad(x, FLOAT32)));
                for (int t = 0; t < tail; t++)
                    out[base + c + t] = y[t];
            }
        }
    } else {
        for (int o = start; o < end; o++)
            for (int c = 0; c < biasSize; c++) {
                Real bc = bias[c];
                size_t base = ((size_t)o * biasSize + c) * inner;
                int i = 0;
                Vector bcv = VectorBcast(bc);
                for (; i + VECTOR_DIMS <= inner; i += VECTOR_DIMS)
                    VectorStore(out + base + i, fn(VectorAdd(VectorLoad(A + base + i, FLOAT32), bcv)));
                if (i < inner) {
                    Real x[VECTOR_DIMS], y[VECTOR_DIMS];
                    int tail = inner - i;
                    for (int t = 0; t < VECTOR_DIMS; t++)
                        x[t] = t < tail ? A[base + i + t] + bc : 0.0f;
                    VectorStore(y, fn(VectorLoad(x, FLOAT32)));
                    for (int t = 0; t < tail; t++)
                        out[base + i + t] = y[t];
                }
            }
    }
}

// dA += dout * f'(A + bias)
static void affineBackwardDa(void *arg, int start, int end)
{
    Operation *op = arg;
    AffineContext *ctx = op->context;
    Real *A = TensorEntries(op->A), *dA = TensorGradients(op->A), *dout = TensorGradients(op->out);
    int biasSize = TensorSize(op->B), inner = ctx->inner;
    VectorUnary dafn = (VectorUnary)ctx->u.da;
    
    // Identity case since it's derivative is 1 regardless of A
    if (ctx->u.f == IDENTITY.f) {
        size_t lo = (size_t)start * biasSize * inner;
        size_t hi = (size_t)end   * biasSize * inner;
        size_t i  = lo;
        for (; i + VECTOR_DIMS <= hi; i += VECTOR_DIMS)
            VectorStore(dA + i, VectorAdd(VectorLoad(dA + i, FLOAT32), VectorLoad(dout + i, FLOAT32)));
        for (; i < hi; i++)
            dA[i] += dout[i];
        return;
    }

    Real bias[biasSize];
    for (int i = 0; i < biasSize; i++) bias[i] = TensorGet(op->B, i);
    if (TensorDataType(op->A) != FLOAT32 || ScaleGet(op->A).values) {
        for (int o = start; o < end; o++)
            for (int c = 0; c < biasSize; c++) {
                Real bc = bias[c];
                size_t base = ((size_t)o * biasSize + c) * inner;
                for (int i = 0; i < inner; i++)
                    dA[base + i] += dout[base + i] * VectorFirst(dafn(VectorBcast(TensorGet(op->A, (int)(base + i)) + bc)));
            }
        return;
    }

    if (inner == 1) {
        for (int o = start; o < end; o++) {
            size_t base = (size_t)o * biasSize;
            int c = 0;
            for (; c + VECTOR_DIMS <= biasSize; c += VECTOR_DIMS) {
                Vector v = VectorAdd(VectorLoad(A + base + c, FLOAT32), VectorLoad(bias + c, FLOAT32));
                VectorStore(dA + base + c, VectorFma(VectorLoad(dout + base + c, FLOAT32), dafn(v), VectorLoad(dA + base + c, FLOAT32)));
            }
            if (c < biasSize) {
                Real x[VECTOR_DIMS], g[VECTOR_DIMS], old[VECTOR_DIMS], y[VECTOR_DIMS];
                int tail = biasSize - c;
                for (int t = 0; t < VECTOR_DIMS; t++) {
                    x[t] = t < tail ? A[base + c + t] + bias[c + t] : 0.0f;
                    g[t] = t < tail ? dout[base + c + t] : 0.0f;
                    old[t] = t < tail ? dA[base + c + t] : 0.0f;
                }
                VectorStore(y, VectorFma(VectorLoad(g, FLOAT32), dafn(VectorLoad(x, FLOAT32)), VectorLoad(old, FLOAT32)));
                for (int t = 0; t < tail; t++)
                    dA[base + c + t] = y[t];
            }
        }
    } else {
        for (int o = start; o < end; o++)
            for (int c = 0; c < biasSize; c++) {
                Real bc = bias[c];
                size_t base = ((size_t)o * biasSize + c) * inner;
                int i = 0;
                Vector bcv = VectorBcast(bc);
                for (; i + VECTOR_DIMS <= inner; i += VECTOR_DIMS) {
                    Vector v = VectorAdd(VectorLoad(A + base + i, FLOAT32), bcv);
                    VectorStore(dA + base + i, VectorFma(VectorLoad(dout + base + i, FLOAT32), dafn(v), VectorLoad(dA + base + i, FLOAT32)));
                }
                if (i < inner) {
                    Real x[VECTOR_DIMS], g[VECTOR_DIMS], old[VECTOR_DIMS], y[VECTOR_DIMS];
                    int tail = inner - i;
                    for (int t = 0; t < VECTOR_DIMS; t++) {
                        x[t] = t < tail ? A[base + i + t] + bc : 0.0f;
                        g[t] = t < tail ? dout[base + i + t] : 0.0f;
                        old[t] = t < tail ? dA[base + i + t] : 0.0f;
                    }
                    VectorStore(y, VectorFma(VectorLoad(g, FLOAT32), dafn(VectorLoad(x, FLOAT32)), VectorLoad(old, FLOAT32)));
                    for (int t = 0; t < tail; t++)
                        dA[base + i + t] = y[t];
                }
            }
    }
}

// dB[c] += sum(dout * f'(A + bias[c]))
static void affineBackwardDb(void *arg, int start, int end)
{
    Operation *op = arg;
    AffineContext *ctx = op->context;
    Real *A = TensorEntries(op->A), *dout = TensorGradients(op->out), *dB = TensorGradients(op->B);
    int biasSize = TensorSize(op->B), inner = ctx->inner;
    int outer = TensorSize(op->out) / (biasSize * inner);
    VectorUnary dafn = (VectorUnary)ctx->u.da;

    // Identity case since it's derivative is 1 regardless of A
    if (ctx->u.f == IDENTITY.f && inner == 1) {
        int c = start;
        for (; c + VECTOR_DIMS <= end; c += VECTOR_DIMS) {
            Vector acc = VectorZero();
            for (int o = 0; o < outer; o++)
                acc = VectorAdd(acc, VectorLoad(dout + (size_t)o * biasSize + c, FLOAT32));
            VectorStore(dB + c, VectorAdd(VectorLoad(dB + c, FLOAT32), acc));
        }
        for (; c < end; c++) {
            Real s = 0;
            for (int o = 0; o < outer; o++)
                s += dout[(size_t)o * biasSize + c];
            dB[c] += s;
        }
        return;
    }

    Real bias[biasSize];
    for (int i = 0; i < biasSize; i++) bias[i] = TensorGet(op->B, i);

    if (TensorDataType(op->A) != FLOAT32 || ScaleGet(op->A).values) {
        for (int c = start; c < end; c++) {
            Real sum = 0, bc = bias[c];
            for (int o = 0; o < outer; o++) {
                size_t base = ((size_t)o * biasSize + c) * inner;
                for (int i = 0; i < inner; i++)
                    sum += dout[base + i] * VectorFirst(dafn(VectorBcast(TensorGet(op->A, (int)(base + i)) + bc)));
            }
            dB[c] += sum;
        }
        return;
    }

    // Fast path for linear layer
    if (inner == 1) {
        int c = start;
        while (c + VECTOR_DIMS <= end) {
            int nblk = 0, cc = c;
            while (nblk < 4 && cc + VECTOR_DIMS <= end) { nblk++; cc += VECTOR_DIMS; }
            Vector acc[4 * VECTOR_DIMS / VECTOR_DIMS];
            for (int j = 0; j < nblk; j++) acc[j] = VectorZero();
            for (int o = 0; o < outer; o++) {
                const Real *ar = A + (size_t)o * biasSize;
                const Real *dr = dout + (size_t)o * biasSize;
                for (int j = 0; j < nblk; j++) {
                    int cj   = c + j * VECTOR_DIMS;
                    Vector z = VectorAdd(VectorLoad(ar + cj, FLOAT32), VectorLoad(bias + cj, FLOAT32));
                    acc[j]   = VectorFma(VectorLoad(dr + cj, FLOAT32), dafn(z), acc[j]);
                }
            }
            for (int j = 0; j < nblk; j++) {
                int cj = c + j * VECTOR_DIMS;
                VectorStore(dB + cj, VectorAdd(VectorLoad(dB + cj, FLOAT32), acc[j]));
            }
            c = cc;
        }
        for (; c < end; c++) {
            Real s = 0, bc = bias[c];
            for (int o = 0; o < outer; o++) {
                Real x[VECTOR_DIMS] = {0}, g[VECTOR_DIMS] = {0}, y[VECTOR_DIMS];
                x[0] = A[(size_t)o * biasSize + c] + bc;
                g[0] = dout[(size_t)o * biasSize + c];
                VectorStore(y, VectorMul(VectorLoad(g, FLOAT32), dafn(VectorLoad(x, FLOAT32))));
                s += y[0];
            }
            dB[c] += s;
        }
        return;
    }

    for (int c = start; c < end; c++) {
        Real s = 0, bc = bias[c];
        Vector sv = VectorZero();
        for (int o = 0; o < outer; o++) {
            size_t base = ((size_t)o * biasSize + c) * inner;
            int i = 0;
            Vector bcv = VectorBcast(bc);
            for (; i + VECTOR_DIMS <= inner; i += VECTOR_DIMS) {
                Vector v = VectorAdd(VectorLoad(A + base + i, FLOAT32), bcv);
                sv = VectorFma(VectorLoad(dout + base + i, FLOAT32), dafn(v), sv);
            }
            if (i < inner) {
                Real x[VECTOR_DIMS], g[VECTOR_DIMS], y[VECTOR_DIMS];
                int tail = inner - i;
                for (int t = 0; t < VECTOR_DIMS; t++) {
                    x[t] = t < tail ? A[base + i + t] + bc : 0.0f;
                    g[t] = t < tail ? dout[base + i + t] : 0.0f;
                }
                VectorStore(y, VectorMul(VectorLoad(g, FLOAT32), dafn(VectorLoad(x, FLOAT32))));
                for (int t = 0; t < tail; t++)
                    s += y[t];
            }
        }
        s += VectorHsum(sv);
        dB[c] += s;
    }
}
