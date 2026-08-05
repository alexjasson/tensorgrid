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
    int autograd = A.gradients != NULL || bias.gradients != NULL;
    Tensor out = TensorNew(A.dimensions, A.shape, autograd);
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
    if (op->A.gradients)
        OperationDispatch(affineBackwardDa, op, outer, TensorSize(op->out));
    if (op->B.gradients)
        OperationDispatch(affineBackwardDb, op, biasSize, TensorSize(op->out));
}


// out = f(A + bias)
static void affineForward(void *arg, int start, int end)
{
    Operation *op = arg;
    AffineContext *ctx = op->context;
    Real *out = op->out.entries, *A = op->A.entries, *bias = op->B.entries;
    int biasSize = TensorSize(op->B), inner = ctx->inner;
    VectorUnary fn = (VectorUnary)ctx->u.f;
    if (inner == 1) {
        for (int o = start; o < end; o++) {
            size_t base = (size_t)o * biasSize;
            int c = 0;
            for (; c + VECTOR_DIMS <= biasSize; c += VECTOR_DIMS)
                VectorStore(out + base + c, fn(VectorAdd(VectorLoad(A + base + c), VectorLoad(bias + c))));
            if (c < biasSize) {
                Real x[VECTOR_DIMS], y[VECTOR_DIMS];
                int tail = biasSize - c;
                for (int t = 0; t < VECTOR_DIMS; t++)
                    x[t] = t < tail ? A[base + c + t] + bias[c + t] : 0.0f;
                VectorStore(y, fn(VectorLoad(x)));
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
                    VectorStore(out + base + i, fn(VectorAdd(VectorLoad(A + base + i), bcv)));
                if (i < inner) {
                    Real x[VECTOR_DIMS], y[VECTOR_DIMS];
                    int tail = inner - i;
                    for (int t = 0; t < VECTOR_DIMS; t++)
                        x[t] = t < tail ? A[base + i + t] + bc : 0.0f;
                    VectorStore(y, fn(VectorLoad(x)));
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
    Real *A = op->A.entries, *bias = op->B.entries, *dA = op->A.gradients, *dout = op->out.gradients;
    int biasSize = TensorSize(op->B), inner = ctx->inner;
    VectorUnary dafn = (VectorUnary)ctx->u.da;
    
    // Identity case since it's derivative is 1 regardless of A
    if (ctx->u.f == IDENTITY.f) {
        size_t lo = (size_t)start * biasSize * inner;
        size_t hi = (size_t)end   * biasSize * inner;
        size_t i  = lo;
        for (; i + VECTOR_DIMS <= hi; i += VECTOR_DIMS)
            VectorStore(dA + i, VectorAdd(VectorLoad(dA + i), VectorLoad(dout + i)));
        for (; i < hi; i++)
            dA[i] += dout[i];
        return;
    }
    if (inner == 1) {
        for (int o = start; o < end; o++) {
            size_t base = (size_t)o * biasSize;
            int c = 0;
            for (; c + VECTOR_DIMS <= biasSize; c += VECTOR_DIMS) {
                Vector v = VectorAdd(VectorLoad(A + base + c), VectorLoad(bias + c));
                VectorStore(dA + base + c, VectorFma(VectorLoad(dout + base + c), dafn(v), VectorLoad(dA + base + c)));
            }
            if (c < biasSize) {
                Real x[VECTOR_DIMS], g[VECTOR_DIMS], old[VECTOR_DIMS], y[VECTOR_DIMS];
                int tail = biasSize - c;
                for (int t = 0; t < VECTOR_DIMS; t++) {
                    x[t] = t < tail ? A[base + c + t] + bias[c + t] : 0.0f;
                    g[t] = t < tail ? dout[base + c + t] : 0.0f;
                    old[t] = t < tail ? dA[base + c + t] : 0.0f;
                }
                VectorStore(y, VectorFma(VectorLoad(g), dafn(VectorLoad(x)), VectorLoad(old)));
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
                    Vector v = VectorAdd(VectorLoad(A + base + i), bcv);
                    VectorStore(dA + base + i, VectorFma(VectorLoad(dout + base + i), dafn(v), VectorLoad(dA + base + i)));
                }
                if (i < inner) {
                    Real x[VECTOR_DIMS], g[VECTOR_DIMS], old[VECTOR_DIMS], y[VECTOR_DIMS];
                    int tail = inner - i;
                    for (int t = 0; t < VECTOR_DIMS; t++) {
                        x[t] = t < tail ? A[base + i + t] + bc : 0.0f;
                        g[t] = t < tail ? dout[base + i + t] : 0.0f;
                        old[t] = t < tail ? dA[base + i + t] : 0.0f;
                    }
                    VectorStore(y, VectorFma(VectorLoad(g), dafn(VectorLoad(x)), VectorLoad(old)));
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
    Real *A = op->A.entries, *bias = op->B.entries, *dout = op->out.gradients, *dB = op->B.gradients;
    int biasSize = TensorSize(op->B), inner = ctx->inner;
    int outer = TensorSize(op->out) / (biasSize * inner);
    VectorUnary dafn = (VectorUnary)ctx->u.da;

    // Identity case since it's derivative is 1 regardless of A
    if (ctx->u.f == IDENTITY.f && inner == 1) {
        int c = start;
        for (; c + VECTOR_DIMS <= end; c += VECTOR_DIMS) {
            Vector acc = VectorZero();
            for (int o = 0; o < outer; o++)
                acc = VectorAdd(acc, VectorLoad(dout + (size_t)o * biasSize + c));
            VectorStore(dB + c, VectorAdd(VectorLoad(dB + c), acc));
        }
        for (; c < end; c++) {
            Real s = 0;
            for (int o = 0; o < outer; o++)
                s += dout[(size_t)o * biasSize + c];
            dB[c] += s;
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
                    Vector z = VectorAdd(VectorLoad(ar + cj), VectorLoad(bias + cj));
                    acc[j]   = VectorFma(VectorLoad(dr + cj), dafn(z), acc[j]);
                }
            }
            for (int j = 0; j < nblk; j++) {
                int cj = c + j * VECTOR_DIMS;
                VectorStore(dB + cj, VectorAdd(VectorLoad(dB + cj), acc[j]));
            }
            c = cc;
        }
        for (; c < end; c++) {
            Real s = 0, bc = bias[c];
            for (int o = 0; o < outer; o++) {
                Real x[VECTOR_DIMS] = {0}, g[VECTOR_DIMS] = {0}, y[VECTOR_DIMS];
                x[0] = A[(size_t)o * biasSize + c] + bc;
                g[0] = dout[(size_t)o * biasSize + c];
                VectorStore(y, VectorMul(VectorLoad(g), dafn(VectorLoad(x))));
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
                Vector v = VectorAdd(VectorLoad(A + base + i), bcv);
                sv = VectorFma(VectorLoad(dout + base + i), dafn(v), sv);
            }
            if (i < inner) {
                Real x[VECTOR_DIMS], g[VECTOR_DIMS], y[VECTOR_DIMS];
                int tail = inner - i;
                for (int t = 0; t < VECTOR_DIMS; t++) {
                    x[t] = t < tail ? A[base + i + t] + bc : 0.0f;
                    g[t] = t < tail ? dout[base + i + t] : 0.0f;
                }
                VectorStore(y, VectorMul(VectorLoad(g), dafn(VectorLoad(x))));
                for (int t = 0; t < tail; t++)
                    s += y[t];
            }
        }
        s += VectorHsum(sv);
        dB[c] += s;
    }
}
