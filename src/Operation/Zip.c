// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Vector.h"

typedef struct { Binary b; } ZipContext;

static void OperationZipForward(Operation *op);
static void OperationZipBackward(Operation *op);
static void zipForward(void *arg, int start, int end);
static void zipBackward(void *arg, int start, int end);

Tensor OperationZip(Binary fn, Tensor A, Tensor B)
{
    Tensor zipA = A;
    Tensor zipB = B;

    // Right-align shapes
    int outDims = A.dimensions >= B.dimensions ? A.dimensions : B.dimensions;
    Tensor *inputs[2] = { &zipA, &zipB };
    for (int n = 0; n < 2; n++) {
        Tensor *t = inputs[n];
        int pad = outDims - t->dimensions;
        for (int i = outDims - 1; i >= pad; i--) {
            t->shape[i]   = t->shape[i - pad];
            t->strides[i] = t->strides[i - pad];
        }
        for (int i = pad - 1; i >= 0; i--) {
            t->shape[i]   = 1;
            t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
        }
        t->dimensions = outDims;
    }

    int outShape[MAX_DIMENSIONS];
    for (int axis = 0; axis < outDims; axis++) {
        int aSize = zipA.shape[axis];
        int bSize = zipB.shape[axis];
        if (aSize != bSize && aSize != 1 && bSize != 1) {
            fprintf(stderr, "TensorZip shapes are not compatible!\n");
            exit(EXIT_FAILURE);
        }
        outShape[axis] = aSize >= bSize ? aSize : bSize;
    }

    for (int axis = 0; axis < outDims; axis++) {
        if (zipA.shape[axis] != outShape[axis]) { zipA.strides[axis] = 0; zipA.shape[axis] = outShape[axis]; }
        if (zipB.shape[axis] != outShape[axis]) { zipB.strides[axis] = 0; zipB.shape[axis] = outShape[axis]; }
    }

    int autograd = zipA.gradients != NULL || zipB.gradients != NULL;
    Tensor out = TensorNew(outDims, outShape, autograd);

    ZipContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->b = fn;
    return OperationExecute(OperationNew(out, zipA, zipB, OperationZipForward, OperationZipBackward, ctx));
}

static void OperationZipForward(Operation *op)
{
    OperationDispatch(zipForward, op, TensorSize(op->out), TensorSize(op->out));
}

static void OperationZipBackward(Operation *op)
{
    if (!op->A.gradients && !op->B.gradients) return;
    OperationDispatch(zipBackward, op, TensorSize(op->out), TensorSize(op->out));
}


static void zipForward(void *arg, int start, int end)
{
    Operation *op  = arg;
    ZipContext *ctx = op->context;
    Real     *out = op->out.entries;
    Real     *A   = op->A.entries;
    Real     *B   = op->B.entries;
    VectorBinary fn = (VectorBinary)ctx->b.fv;
    int aC = isContiguous(op->A.shape, op->A.strides, op->A.dimensions);
    int bC = isContiguous(op->B.shape, op->B.strides, op->B.dimensions);

    // Contiguous fast path
    if (aC && bC) {
        int i = start;
        for (; i + VECTOR_DIMS <= end; i += VECTOR_DIMS)
            VectorStore(out + i, fn(VectorLoad(A + i), VectorLoad(B + i)));
        if (i < end) {
            Real x[VECTOR_DIMS], y[VECTOR_DIMS], z[VECTOR_DIMS];
            int tail = end - i;
            for (int t = 0; t < VECTOR_DIMS; t++) {
                x[t] = t < tail ? A[i + t] : 1.0f;
                y[t] = t < tail ? B[i + t] : 1.0f;
            }
            VectorStore(z, fn(VectorLoad(x), VectorLoad(y)));
            for (int t = 0; t < tail; t++)
                out[i + t] = z[t];
        }
        return;
    }
    // Broadcast aware path
    int dims = op->out.dimensions;
    const int *shape = op->out.shape;
    const int *sA = op->A.strides, *sB = op->B.strides;
    int clsA = -1, clsB = -1, runAxes = 0, R = 1;
    for (int ax = dims - 1; ax >= 0; ax--) {
        if (shape[ax] == 1) { runAxes++; continue; }
        int okA, okB;
        if (clsA == -1) {
            if (sA[ax] == 1)      { clsA = 1; okA = 1; }
            else if (sA[ax] == 0) { clsA = 0; okA = 1; }
            else                    okA = 0;
        } else okA = (clsA == 1) ? (sA[ax] == R) : (sA[ax] == 0);
        if (clsB == -1) {
            if (sB[ax] == 1)      { clsB = 1; okB = 1; }
            else if (sB[ax] == 0) { clsB = 0; okB = 1; }
            else                    okB = 0;
        } else okB = (clsB == 1) ? (sB[ax] == R) : (sB[ax] == 0);
        if (!okA || !okB) break;
        R *= shape[ax];
        runAxes++;
    }
    if (clsA >= 0 && clsB >= 0 && R >= VECTOR_DIMS) {
        int outerDims = dims - runAxes;
        int oidx[MAX_DIMENSIONS];
        int b0 = start / R, r0 = start - b0 * R, rem = b0;
        long offA = 0, offB = 0;
        for (int ax = outerDims - 1; ax >= 0; ax--) {
            int idx = rem % shape[ax]; rem /= shape[ax];
            oidx[ax] = idx;
            offA += (long)idx * sA[ax];
            offB += (long)idx * sB[ax];
        }
        int o = start, r = r0;
        while (o < end) {
            int len = R - r;
            if (o + len > end) len = end - o;
            Real       *po = out + o;
            const Real *pa = A + offA + (clsA ? r : 0);
            const Real *pb = B + offB + (clsB ? r : 0);
            int i = 0;
            if (clsA && clsB) {
                for (; i + VECTOR_DIMS <= len; i += VECTOR_DIMS)
                    VectorStore(po + i, fn(VectorLoad(pa + i), VectorLoad(pb + i)));
            } else if (clsA) {
                Vector vb = VectorBcast(*pb);
                for (; i + VECTOR_DIMS <= len; i += VECTOR_DIMS)
                    VectorStore(po + i, fn(VectorLoad(pa + i), vb));
            } else if (clsB) {
                Vector va = VectorBcast(*pa);
                for (; i + VECTOR_DIMS <= len; i += VECTOR_DIMS)
                    VectorStore(po + i, fn(va, VectorLoad(pb + i)));
            } else {
                Vector v = fn(VectorBcast(*pa), VectorBcast(*pb));
                for (; i + VECTOR_DIMS <= len; i += VECTOR_DIMS)
                    VectorStore(po + i, v);
            }
            if (i < len) {
                Real x[VECTOR_DIMS], y[VECTOR_DIMS], z[VECTOR_DIMS];
                int tail = len - i;
                for (int t = 0; t < VECTOR_DIMS; t++) {
                    x[t] = clsA ? (t < tail ? pa[i + t] : 1.0f) : *pa;
                    y[t] = clsB ? (t < tail ? pb[i + t] : 1.0f) : *pb;
                }
                VectorStore(z, fn(VectorLoad(x), VectorLoad(y)));
                for (int t = 0; t < tail; t++) po[i + t] = z[t];
            }
            o += len; r += len;
            if (r == R) {
                r = 0;
                for (int ax = outerDims - 1; ax >= 0; ax--) {
                    offA += sA[ax]; offB += sB[ax];
                    if (++oidx[ax] < shape[ax]) break;
                    oidx[ax] = 0;
                    offA -= (long)shape[ax] * sA[ax];
                    offB -= (long)shape[ax] * sB[ax];
                }
            }
        }
        return;
    }

    // Default path
    for (int i = start; i < end; i += VECTOR_DIMS) {
        Real x[VECTOR_DIMS], y[VECTOR_DIMS], z[VECTOR_DIMS];
        int tail = end - i < VECTOR_DIMS ? end - i : VECTOR_DIMS;
        for (int t = 0; t < VECTOR_DIMS; t++) {
            int j = i + t;
            x[t] = t < tail ? A[aC ? j : unravel(j, op->A.shape, op->A.strides, op->A.dimensions)] : 1.0f;
            y[t] = t < tail ? B[bC ? j : unravel(j, op->B.shape, op->B.strides, op->B.dimensions)] : 1.0f;
        }
        VectorStore(z, fn(VectorLoad(x), VectorLoad(y)));
        for (int t = 0; t < tail; t++)
            out[i + t] = z[t];
    }
}

static void zipBackward(void *arg, int start, int end)
{
    Operation  *op = arg;

    // Finds the optimal start/end values so each thread gets a multiple of G work
    // and the last thread doesn't do extra work
    int total = TensorSize(op->out);
    if (start != 0 || end != total) {
        int G = 1;
        Tensor *in[2] = { &op->A, &op->B };
        for (int n = 0; n < 2; n++) {
            if (!in[n]->gradients) continue;
            for (int ax = 0; ax < op->out.dimensions; ax++)
                if (in[n]->strides[ax] == 0 && in[n]->shape[ax] > 1) {
                    int g = 1;
                    for (int d = ax; d < op->out.dimensions; d++)
                        g *= op->out.shape[d];
                    if (g > G) G = g;
                    break;
                }
        }
        if (G > 1) {
            start = (start + G - 1) / G * G;
            end   = end == total ? total : (end + G - 1) / G * G;
            if (start >= end) return;
        }
    }

    ZipContext *ctx = op->context;
    Real       *A = op->A.entries;
    Real       *B = op->B.entries;
    Real      *dA = op->A.gradients;
    Real      *dB = op->B.gradients;
    Real    *dout = op->out.gradients;
    VectorBinary dafn = (VectorBinary)ctx->b.dav;
    VectorBinary dbfn = (VectorBinary)ctx->b.dbv;
    int aC = isContiguous(op->A.shape, op->A.strides, op->A.dimensions);
    int bC = isContiguous(op->B.shape, op->B.strides, op->B.dimensions);

    // Contiguous fast path
    if (aC && bC) {
        int j = start;
        for (; j + VECTOR_DIMS <= end; j += VECTOR_DIMS) {
            Vector a = VectorLoad(A + j);
            Vector b = VectorLoad(B + j);
            Vector g = VectorLoad(dout + j);
            if (dA) VectorStore(dA + j, VectorFma(g, dafn(a, b), VectorLoad(dA + j)));
            if (dB) VectorStore(dB + j, VectorFma(g, dbfn(a, b), VectorLoad(dB + j)));
        }
        if (j < end) {
            Real x[VECTOR_DIMS], y[VECTOR_DIMS], g[VECTOR_DIMS], r[VECTOR_DIMS], old[VECTOR_DIMS];
            int tail = end - j;
            for (int t = 0; t < VECTOR_DIMS; t++) {
                x[t] = t < tail ? A[j + t] : 1.0f;
                y[t] = t < tail ? B[j + t] : 1.0f;
                g[t] = t < tail ? dout[j + t] : 0.0f;
            }
            if (dA) {
                for (int t = 0; t < VECTOR_DIMS; t++)
                    old[t] = t < tail ? dA[j + t] : 0.0f;
                VectorStore(r, VectorFma(VectorLoad(g), dafn(VectorLoad(x), VectorLoad(y)), VectorLoad(old)));
                for (int t = 0; t < tail; t++)
                    dA[j + t] = r[t];
            }
            if (dB) {
                for (int t = 0; t < VECTOR_DIMS; t++)
                    old[t] = t < tail ? dB[j + t] : 0.0f;
                VectorStore(r, VectorFma(VectorLoad(g), dbfn(VectorLoad(x), VectorLoad(y)), VectorLoad(old)));
                for (int t = 0; t < tail; t++)
                    dB[j + t] = r[t];
            }
        }
        return;
    }

    // Broadcast aware path
    int dims = op->out.dimensions;
    const int *shape = op->out.shape;
    const int *sA = op->A.strides, *sB = op->B.strides;
    int clsA = -1, clsB = -1, runAxes = 0, R = 1;
    for (int ax = dims - 1; ax >= 0; ax--) {
        if (shape[ax] == 1) { runAxes++; continue; }
        int okA, okB;
        if (clsA == -1) {
            if (sA[ax] == 1)      { clsA = 1; okA = 1; }
            else if (sA[ax] == 0) { clsA = 0; okA = 1; }
            else                    okA = 0;
        } else okA = (clsA == 1) ? (sA[ax] == R) : (sA[ax] == 0);
        if (clsB == -1) {
            if (sB[ax] == 1)      { clsB = 1; okB = 1; }
            else if (sB[ax] == 0) { clsB = 0; okB = 1; }
            else                    okB = 0;
        } else okB = (clsB == 1) ? (sB[ax] == R) : (sB[ax] == 0);
        if (!okA || !okB) break;
        R *= shape[ax];
        runAxes++;
    }
    if (clsA >= 0 && clsB >= 0 && R >= VECTOR_DIMS) {
        int outerDims = dims - runAxes;
        int oidx[MAX_DIMENSIONS];
        int b0 = start / R, r0 = start - b0 * R, rem = b0;
        long offA = 0, offB = 0;
        for (int ax = outerDims - 1; ax >= 0; ax--) {
            int idx = rem % shape[ax]; rem /= shape[ax];
            oidx[ax] = idx;
            offA += (long)idx * sA[ax];
            offB += (long)idx * sB[ax];
        }
        int o = start, r = r0;
        while (o < end) {
            int len = R - r;
            if (o + len > end) len = end - o;
            const Real *pa  = A + offA + (clsA ? r : 0);
            const Real *pb  = B + offB + (clsB ? r : 0);
            const Real *pg  = dout + o;
            Real       *pdA = dA ? dA + offA + (clsA ? r : 0) : NULL;
            Real       *pdB = dB ? dB + offB + (clsB ? r : 0) : NULL;
            Vector accA = VectorZero(), accB = VectorZero();
            Vector va = clsA ? VectorZero() : VectorBcast(*pa);
            Vector vb = clsB ? VectorZero() : VectorBcast(*pb);
            int i = 0;
            for (; i + VECTOR_DIMS <= len; i += VECTOR_DIMS) {
                Vector av = clsA ? VectorLoad(pa + i) : va;
                Vector bv = clsB ? VectorLoad(pb + i) : vb;
                Vector g  = VectorLoad(pg + i);
                if (pdA) {
                    Vector d = VectorMul(g, dafn(av, bv));
                    if (clsA) VectorStore(pdA + i, VectorAdd(VectorLoad(pdA + i), d));
                    else      accA = VectorAdd(accA, d);
                }
                if (pdB) {
                    Vector d = VectorMul(g, dbfn(av, bv));
                    if (clsB) VectorStore(pdB + i, VectorAdd(VectorLoad(pdB + i), d));
                    else      accB = VectorAdd(accB, d);
                }
            }
            if (i < len) {
                Real x[VECTOR_DIMS], y[VECTOR_DIMS], gg[VECTOR_DIMS];
                int tail = len - i;
                for (int t = 0; t < VECTOR_DIMS; t++) {
                    x[t]  = clsA ? (t < tail ? pa[i + t] : 0.0f) : *pa;
                    y[t]  = clsB ? (t < tail ? pb[i + t] : 0.0f) : *pb;
                    gg[t] = t < tail ? pg[i + t] : 0.0f;
                }
                Vector av = VectorLoad(x), bv = VectorLoad(y), g = VectorLoad(gg);
                if (pdA) {
                    Vector d = VectorMul(g, dafn(av, bv));
                    if (clsA) { Real tmp[VECTOR_DIMS]; VectorStore(tmp, d);
                                for (int t = 0; t < tail; t++) pdA[i + t] += tmp[t]; }
                    else accA = VectorAdd(accA, d);
                }
                if (pdB) {
                    Vector d = VectorMul(g, dbfn(av, bv));
                    if (clsB) { Real tmp[VECTOR_DIMS]; VectorStore(tmp, d);
                                for (int t = 0; t < tail; t++) pdB[i + t] += tmp[t]; }
                    else accB = VectorAdd(accB, d);
                }
            }
            if (pdA && !clsA) pdA[0] += VectorHsum(accA);
            if (pdB && !clsB) pdB[0] += VectorHsum(accB);
            o += len; r += len;
            if (r == R) {
                r = 0;
                for (int ax = outerDims - 1; ax >= 0; ax--) {
                    offA += sA[ax]; offB += sB[ax];
                    if (++oidx[ax] < shape[ax]) break;
                    oidx[ax] = 0;
                    offA -= (long)shape[ax] * sA[ax];
                    offB -= (long)shape[ax] * sB[ax];
                }
            }
        }
        return;
    }

    // Default path
    for (int j = start; j < end; j += VECTOR_DIMS) {
        Real x[VECTOR_DIMS], y[VECTOR_DIMS], g[VECTOR_DIMS], r[VECTOR_DIMS];
        int aOffs[VECTOR_DIMS], bOffs[VECTOR_DIMS];
        int tail = end - j < VECTOR_DIMS ? end - j : VECTOR_DIMS;
        for (int t = 0; t < VECTOR_DIMS; t++) {
            int i = j + t;
            aOffs[t] = 0;
            bOffs[t] = 0;
            x[t] = 1.0f;
            y[t] = 1.0f;
            g[t] = 0.0f;
            if (t < tail) {
                aOffs[t] = aC ? i : unravel(i, op->A.shape, op->A.strides, op->A.dimensions);
                bOffs[t] = bC ? i : unravel(i, op->B.shape, op->B.strides, op->B.dimensions);
                x[t] = A[aOffs[t]];
                y[t] = B[bOffs[t]];
                g[t] = dout[i];
            }
        }
        if (dA) {
            VectorStore(r, VectorMul(VectorLoad(g), dafn(VectorLoad(x), VectorLoad(y))));
            for (int t = 0; t < tail; t++)
                dA[aOffs[t]] += r[t];
        }
        if (dB) {
            VectorStore(r, VectorMul(VectorLoad(g), dbfn(VectorLoad(x), VectorLoad(y))));
            for (int t = 0; t < tail; t++)
                dB[bOffs[t]] += r[t];
        }
    }
}
