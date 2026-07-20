// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Scalar.h"
#include "../math/Vector.h"

typedef struct {
    int    rows, D, contiguous;
    Real   eps;
    Real *mean;   // [rows], cached in forward for backward
    Real *rstd;   // [rows]
} NormContext;

static void OperationNormalizeForward(Operation *op);
static void OperationNormalizeBackward(Operation *op);
static void normForward(void *arg, int start, int end);
static void normBackward(void *arg, int start, int end);

Tensor OperationNormalize(Tensor in, Real eps)
{
    int D    = TensorShape(in)[TensorDimensions(in) - 1];
    int rows = TensorSize(in) / D;

    NormContext *ctx = malloc(sizeof(NormContext) + (size_t)2 * rows * sizeof(Real));
    if (!ctx) { fprintf(stderr, "Insufficient memory!\n"); exit(EXIT_FAILURE); }
    ctx->rows       = rows;
    ctx->D          = D;
    ctx->contiguous = isContiguous(TensorShape(in), TensorStride(in), TensorDimensions(in));
    ctx->eps        = eps;
    ctx->mean       = (Real *)(ctx + 1);
    ctx->rstd       = ctx->mean + rows;

    Tensor out = TensorNew(TensorDimensions(in), TensorShape(in), REAL, TensorGradients(in) != NULL);
    return OperationExecute(OperationNew(out, in, NULL, OperationNormalizeForward, OperationNormalizeBackward, ctx));
}

static void OperationNormalizeForward(Operation *op)
{
    NormContext *ctx = op->context;
    OperationDispatch(normForward, op, ctx->rows, (long)ctx->rows * ctx->D);
}

static void OperationNormalizeBackward(Operation *op)
{
    if (!TensorGradients(op->A)) return;
    NormContext *ctx = op->context;
    OperationDispatch(normBackward, op, ctx->rows, (long)ctx->rows * ctx->D);
}

static void normForward(void *arg, int start, int end)
{
    Operation   *op  = arg;
    NormContext *ctx = op->context;
    int          D   = ctx->D;
    Tensor       A   = op->A;
    Real        *in  = TensorEntries(A);
    Real        *out = TensorEntries(op->out);
    int          sD  = TensorStride(A)[TensorDimensions(A) - 1];

    if (TensorDataType(A) != REAL || TensorScale(A)) {
        for (int r = start; r < end; r++) {
            int xBase = ctx->contiguous ? r * D : unravel(r, TensorShape(A), TensorStride(A), TensorDimensions(A) - 1);
            Real *o = out + (size_t)r * D;

            Real sum = 0.0f;
            for (int i = 0; i < D; i++)
                sum += TensorGet(A, xBase + i * sD);
            Real mu = sum / D;

            Real var = 0.0f;
            for (int i = 0; i < D; i++) {
                Real d = TensorGet(A, xBase + i * sD) - mu;
                var += d * d;
            }
            Real rstd = 1.0f / sqrte(var / D + ctx->eps);
            ctx->mean[r] = mu;
            ctx->rstd[r] = rstd;

            for (int i = 0; i < D; i++)
                o[i] = (TensorGet(A, xBase + i * sD) - mu) * rstd;
        }
        return;
    }

    for (int r = start; r < end; r++) {
        int xBase = ctx->contiguous ? r * D : unravel(r, TensorShape(A), TensorStride(A), TensorDimensions(A) - 1);
        const Real *x = in + xBase;
        Real       *o = out + (size_t)r * D;

        Real sum = 0.0f;
        int i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vs = VectorZero();
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) vs = VectorAdd(vs, VectorLoad(x + i, REAL));
            sum = VectorHsum(vs);
            for (; i < D; i++) sum += x[i];
        } else {
            for (; i < D; i++) sum += x[i * sD];
        }
        Real mu = sum / D;

        Real var = 0.0f;
        i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vv = VectorZero();
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) {
                Vector d = VectorSub(VectorLoad(x + i, REAL), vmu);
                vv = VectorFma(d, d, vv);
            }
            var = VectorHsum(vv);
            for (; i < D; i++) { Real d = x[i] - mu; var += d * d; }
        } else {
            for (; i < D; i++) { Real d = x[i * sD] - mu; var += d * d; }
        }
        var /= D;

        Real rstd = 1.0f / sqrte(var + ctx->eps);
        ctx->mean[r] = mu;
        ctx->rstd[r] = rstd;

        i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vr = VectorBcast(rstd);
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS)
                VectorStore(o + i, VectorMul(VectorSub(VectorLoad(x + i, REAL), vmu), vr));
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
    Real        *in   = TensorEntries(A);
    Real        *dx   = TensorGradients(A);
    Real        *dout = TensorGradients(op->out);
    int          sD   = TensorStride(A)[TensorDimensions(A) - 1];

    for (int r = start; r < end; r++) {
        int xBase = ctx->contiguous ? r * D : unravel(r, TensorShape(A), TensorStride(A), TensorDimensions(A) - 1);
        const Real *x  = in + xBase;
        const Real *dy = dout + (size_t)r * D;
        Real       *gx = dx + xBase;
        Real  mu   = ctx->mean[r];
        Real  rstd = ctx->rstd[r];

        Real s1 = 0.0f, s2 = 0.0f;
        int i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vr = VectorBcast(rstd);
            Vector vs1 = VectorZero(), vs2 = VectorZero();
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) {
                Vector dyv = VectorLoad(dy + i, REAL);
                Vector y   = VectorMul(VectorSub(VectorLoad(x + i, REAL), vmu), vr);
                vs1 = VectorAdd(vs1, dyv);
                vs2 = VectorFma(dyv, y, vs2);
            }
            s1 = VectorHsum(vs1);
            s2 = VectorHsum(vs2);
            for (; i < D; i++) { Real y = (x[i] - mu) * rstd; s1 += dy[i]; s2 += dy[i] * y; }
        } else {
            for (; i < D; i++) { Real y = (x[i * sD] - mu) * rstd; s1 += dy[i]; s2 += dy[i] * y; }
        }
        Real c1 = s1 / D, c2 = s2 / D;

        i = 0;
        if (ctx->contiguous || sD == 1) {
            Vector vmu = VectorBcast(mu), vr = VectorBcast(rstd);
            Vector vc1 = VectorBcast(c1), vc2 = VectorBcast(c2);
            for (; i + VECTOR_DIMS <= D; i += VECTOR_DIMS) {
                Vector y   = VectorMul(VectorSub(VectorLoad(x + i, REAL), vmu), vr);
                Vector dyv = VectorLoad(dy + i, REAL);
                Vector res = VectorMul(vr, VectorSub(VectorSub(dyv, vc1), VectorMul(y, vc2)));
                VectorStore(gx + i, VectorAdd(VectorLoad(gx + i, REAL), res));
            }
            for (; i < D; i++) {
                Real y = (x[i] - mu) * rstd;
                gx[i] += rstd * (dy[i] - c1 - y * c2);
            }
        } else {
            for (; i < D; i++) {
                Real y = (x[i * sD] - mu) * rstd;
                gx[i * sD] += rstd * (dy[i] - c1 - y * c2);
            }
        }
    }
}
