// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../Scalar.h"
#include "../Vector.h"

typedef struct {
    int    rows, D, contiguous;
    float  eps;
    Entry *mean;   // [rows], cached in forward for backward
    Entry *rstd;   // [rows]
} NormContext;

static void OperationNormalizeForward(Operation *op);
static void OperationNormalizeBackward(Operation *op);
static void normForward(void *arg, int start, int end);
static void normBackward(void *arg, int start, int end);

Tensor OperationNormalize(Tensor in, float eps)
{
    int D    = in.shape[in.dimensions - 1];
    int rows = TensorSize(in) / D;

    NormContext *ctx = malloc(sizeof(NormContext) + (size_t)2 * rows * sizeof(Entry));
    if (!ctx) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
    ctx->rows       = rows;
    ctx->D          = D;
    ctx->contiguous = isContiguous(in.shape, in.strides, in.dimensions);
    ctx->eps        = eps;
    ctx->mean       = (Entry *)(ctx + 1);
    ctx->rstd       = ctx->mean + rows;

    Tensor out = TensorNew(in.dimensions, in.shape, in.gradients != NULL);
    return OperationExecute(OperationNew(out, in, (Tensor){0}, OperationNormalizeForward, OperationNormalizeBackward, ctx));
}

static void OperationNormalizeForward(Operation *op)
{
    NormContext *ctx = op->context;
    OperationDispatch(normForward, op, ctx->rows, (long)ctx->rows * ctx->D);
}

static void OperationNormalizeBackward(Operation *op)
{
    if (!op->A.gradients) return;
    NormContext *ctx = op->context;
    OperationDispatch(normBackward, op, ctx->rows, (long)ctx->rows * ctx->D);
}

static void normForward(void *arg, int start, int end)
{
    Operation   *op  = arg;
    NormContext *ctx = op->context;
    int          D   = ctx->D;
    Tensor       A   = op->A;
    Entry       *in  = A.entries;
    Entry       *out = op->out.entries;
    int          sD  = A.strides[A.dimensions - 1];

    for (int r = start; r < end; r++) {
        int xBase = ctx->contiguous ? r * D : unravel(r, A.shape, A.strides, A.dimensions - 1);
        const Entry *x = in + xBase;
        Entry       *o = out + (size_t)r * D;

        Entry sum = 0.0f;
        int i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vs = VectorZero();
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) vs = VectorAdd(vs, VectorLoad(x + i));
            sum = VectorHsum(vs);
            for (; i < D; i++) sum += x[i];
        } else {
            for (; i < D; i++) sum += x[i * sD];
        }
        Entry mu = sum / D;

        Entry var = 0.0f;
        i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vv = VectorZero();
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) {
                Vector d = VectorSub(VectorLoad(x + i), vmu);
                vv = VectorFma(d, d, vv);
            }
            var = VectorHsum(vv);
            for (; i < D; i++) { Entry d = x[i] - mu; var += d * d; }
        } else {
            for (; i < D; i++) { Entry d = x[i * sD] - mu; var += d * d; }
        }
        var /= D;

        Entry rstd = 1.0f / sqrte(var + ctx->eps);
        ctx->mean[r] = mu;
        ctx->rstd[r] = rstd;

        i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vr = VectorBcast(rstd);
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS)
                VectorStore(o + i, VectorMul(VectorSub(VectorLoad(x + i), vmu), vr));
            for (; i < D; i++) o[i] = (x[i] - mu) * rstd;
        } else {
            for (; i < D; i++) o[i] = (x[i * sD] - mu) * rstd;
        }
    }
}

static void normBackward(void *arg, int start, int end)
{
    Operation   *op   = arg;
    NormContext *ctx  = op->context;
    int          D    = ctx->D;
    Tensor       A    = op->A;
    Entry       *in   = A.entries;
    Entry       *dx   = A.gradients;
    Entry       *dout = op->out.gradients;
    int          sD   = A.strides[A.dimensions - 1];

    for (int r = start; r < end; r++) {
        int xBase = ctx->contiguous ? r * D : unravel(r, A.shape, A.strides, A.dimensions - 1);
        const Entry *x  = in + xBase;
        const Entry *dy = dout + (size_t)r * D;
        Entry       *gx = dx + xBase;
        Entry  mu   = ctx->mean[r];
        Entry  rstd = ctx->rstd[r];

        Entry s1 = 0.0f, s2 = 0.0f;
        int i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vr = VectorBcast(rstd);
            Vector vs1 = VectorZero(), vs2 = VectorZero();
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) {
                Vector dyv = VectorLoad(dy + i);
                Vector y   = VectorMul(VectorSub(VectorLoad(x + i), vmu), vr);
                vs1 = VectorAdd(vs1, dyv);
                vs2 = VectorFma(dyv, y, vs2);
            }
            s1 = VectorHsum(vs1);
            s2 = VectorHsum(vs2);
            for (; i < D; i++) { Entry y = (x[i] - mu) * rstd; s1 += dy[i]; s2 += dy[i] * y; }
        } else {
            for (; i < D; i++) { Entry y = (x[i * sD] - mu) * rstd; s1 += dy[i]; s2 += dy[i] * y; }
        }
        Entry c1 = s1 / D, c2 = s2 / D;

        i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vr = VectorBcast(rstd);
            Vector vc1 = VectorBcast(c1), vc2 = VectorBcast(c2);
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) {
                Vector y   = VectorMul(VectorSub(VectorLoad(x + i), vmu), vr);
                Vector dyv = VectorLoad(dy + i);
                Vector res = VectorMul(vr, VectorSub(VectorSub(dyv, vc1), VectorMul(y, vc2)));
                VectorStore(gx + i, VectorAdd(VectorLoad(gx + i), res));
            }
            for (; i < D; i++) {
                Entry y = (x[i] - mu) * rstd;
                gx[i] += rstd * (dy[i] - c1 - y * c2);
            }
        } else {
            for (; i < D; i++) {
                Entry y = (x[i * sD] - mu) * rstd;
                gx[i * sD] += rstd * (dy[i] - c1 - y * c2);
            }
        }
    }
}
