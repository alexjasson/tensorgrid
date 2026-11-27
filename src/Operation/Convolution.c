// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>
#include "../Operation.h"
#include "../math/Vector.h"
#include "../utility.h"
#include "../math/Matrix.h"

typedef struct {
    int fH, fW, stride, padding;
} ConvolutionContext;

static void OperationConvolutionForward(Operation *op);
static void OperationConvolutionBackward(Operation *op);
static void convolutionForward(void *arg, int start, int end);
static void convolutionBackwardDb(void *arg, int start, int end);
static void convolutionBackwardDa(void *arg, int start, int end);

static void im2col(Operation *op, Real *patches, int n, int kStart, int kEnd);

Tensor OperationConvolution(Tensor in, Tensor filter, int stride, int padding)
{
    int N    = TensorShape(in)[0];
    int H    = TensorShape(in)[2];
    int W    = TensorShape(in)[3];
    int Cout = TensorShape(filter)[0];
    int fH   = TensorShape(filter)[2];
    int fW   = TensorShape(filter)[3];
    int Hout = (H + 2 * padding - fH) / stride + 1;
    int Wout = (W + 2 * padding - fW) / stride + 1;
    int autograd = TensorGradients(in) != NULL || TensorGradients(filter) != NULL;

    ConvolutionContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->stride = stride;
    ctx->padding = padding;
    ctx->fH = fH;
    ctx->fW = fW;
    Tensor out = TensorNew(4, (int[]){N, Cout, Hout, Wout}, FLOAT32, autograd);
    return OperationExecute(OperationNew(out, in, filter,
                            OperationConvolutionForward, OperationConvolutionBackward, ctx));
}

static void OperationConvolutionForward(Operation *op)
{
    ConvolutionContext *ctx = op->context;
    int Cout = TensorShape(op->B)[0];
    int K    = TensorShape(op->B)[1] * ctx->fH * ctx->fW;
    int S    = TensorShape(op->out)[2] * TensorShape(op->out)[3];
    int N    = TensorShape(op->out)[0];

    OperationDispatch(convolutionForward, op, N, (long)N * Cout * S * K);
}

static void OperationConvolutionBackward(Operation *op)
{
    ConvolutionContext *ctx = op->context;
    int Cout = TensorShape(op->B)[0];
    int K    = TensorShape(op->B)[1] * ctx->fH * ctx->fW;
    int S    = TensorShape(op->out)[2] * TensorShape(op->out)[3];
    int N    = TensorShape(op->out)[0];
    long matmulWork = (long)N * Cout * S * K;

    if (TensorGradients(op->B))
        OperationDispatch(convolutionBackwardDb, op, N, matmulWork);
    if (TensorGradients(op->A))
        OperationDispatch(convolutionBackwardDa, op, N, matmulWork);
}

static void im2col(Operation *op, Real *patches, int n, int kStart, int kEnd)
{
    ConvolutionContext *ctx = op->context;
    Tensor A   = op->A;
    int fH     = ctx->fH,     fW      = ctx->fW;
    int H      = TensorShape(A)[2], W       = TensorShape(A)[3];
    int Hout   = TensorShape(op->out)[2], Wout = TensorShape(op->out)[3];
    int stride = ctx->stride, padding = ctx->padding;
    int sN = TensorStride(A)[0], sC = TensorStride(A)[1], sH = TensorStride(A)[2], sW = TensorStride(A)[3];
    DataType type = TensorDataType(A);
    Scale scale   = ScaleGet(A);
    int real      = type == FLOAT32 && !scale.values;
    int S = Hout * Wout;
    for (int k = kStart; k < kEnd; k++) {
        int ci = k / (fH * fW);
        int fh = (k / fW) % fH;
        int fw = k % fW;
        Real *row = patches + (size_t)(k - kStart) * S;
        int planeOff = n * sN + ci * sC;
        for (int oh = 0; oh < Hout; oh++) {
            Real *outRow = row + (size_t)oh * Wout;
            int ih = oh * stride - padding + fh;
            if (ih < 0 || ih >= H) {
                memset(outRow, 0, (size_t)Wout * sizeof(Real));
                continue;
            }
            int rowOff = planeOff + ih * sH;
            int contiguousRun = stride == 1 && sW == 1;
            int oneGroup = !scale.stride || (rowOff % scale.stride) + W <= scale.stride;
            if (contiguousRun && (real || oneGroup)) {
                int ow0 = padding - fw;     if (ow0 < 0)    ow0 = 0;
                int ow1 = W + padding - fw; if (ow1 > Wout) ow1 = Wout;
                if (ow0 > 0)    memset(outRow, 0, (size_t)ow0 * sizeof(Real));
                size_t from = (size_t)(rowOff + (ow0 - padding + fw));
                if (ow1 > ow0) {
                    if (real) {
                        memcpy(outRow + ow0, (const Real *)TensorEntries(A) + from, (size_t)(ow1 - ow0) * sizeof(Real));
                    } else {
                        Vector s = VectorBcast(ScaleAt(scale, (int)from)), z = VectorBcast(ScaleZeroAt(scale, (int)from));
                        int ow = ow0;
                        for (; ow + VECTOR_DIMS <= ow1; ow += VECTOR_DIMS)
                            VectorStore(outRow + ow, VectorMul(s, VectorSub(
                                VectorLoadAt(TensorEntries(A), from + (size_t)(ow - ow0), type), z)));
                        for (; ow < ow1; ow++)
                            outRow[ow] = TensorGet(A, (int)from + (ow - ow0));
                    }
                }
                if (ow1 < Wout) memset(outRow + ow1, 0, (size_t)(Wout - ow1) * sizeof(Real));
            } else {
                for (int ow = 0; ow < Wout; ow++) {
                    int iw = ow * stride - padding + fw;
                    outRow[ow] = (iw >= 0 && iw < W) ? TensorGet(A, rowOff + iw * sW) : 0.0f;
                }
            }
        }
    }
}

static void convolutionForward(void *arg, int start, int end)
{
    Operation *op = arg;
    ConvolutionContext *ctx = op->context;
    int Cin    = TensorShape(op->B)[1];
    int Cout   = TensorShape(op->B)[0];
    int fH     = ctx->fH, fW = ctx->fW;
    int S      = TensorShape(op->out)[2] * TensorShape(op->out)[3];
    int K      = Cin * fH * fW;
    Real patches[(size_t)K * S];

    Matrix Fv = MatrixView(TensorEntries(op->B), Cout, K, K, 1, TensorDataType(op->B), ScaleGet(op->B));

    for (int n = start; n < end; n++) {
        im2col(op, patches, n, 0, K);
        Real *prod = (Real *)TensorEntries(op->out) + (size_t)n * Cout * S;
        memset(prod, 0, (size_t)Cout * S * sizeof(Real));
        MatrixMultiply(Fv,
                       MatrixView(patches, K, S, S, 1, FLOAT32, (Scale){0}),
                       MatrixView(prod, Cout, S, S, 1, FLOAT32, (Scale){0}));
    }
}

static void convolutionBackwardDb(void *arg, int start, int end)
{
    Operation *op = arg;
    ConvolutionContext *ctx = op->context;
    int Cout   = TensorShape(op->B)[0];
    int S      = TensorShape(op->out)[2] * TensorShape(op->out)[3];
    int K      = TensorShape(op->B)[1] * ctx->fH * ctx->fW;
    Real *dProd   = TensorGradients(op->out);
    Real *dFilter = TensorGradients(op->B);
    Real patches[(size_t)K * S];
    Real local[(size_t)Cout * K];
    memset(local, 0, (size_t)Cout * K * sizeof(Real));

    for (int n = start; n < end; n++) {
        im2col(op, patches, n, 0, K);
        Real *dProd_n = dProd + (size_t)n * Cout * S;
        MatrixMultiply(MatrixView(dProd_n, Cout, S, S, 1, FLOAT32, (Scale){0}),
                       MatrixView(patches, S, K, 1, S, FLOAT32, (Scale){0}),
                       MatrixView(local, Cout, K, K, 1, FLOAT32, (Scale){0}));
    }

    ThreadPool pool = ThreadPoolGet();
    ThreadPoolLock(pool);
    for (size_t i = 0; i < (size_t)Cout * K; i++)
        dFilter[i] += local[i];
    ThreadPoolUnlock(pool);
}

static void convolutionBackwardDa(void *arg, int start, int end)
{
    Operation *op = arg;
    ConvolutionContext *ctx = op->context;
    Tensor A = op->A;
    int Cin  = TensorShape(op->B)[1];
    int Cout = TensorShape(op->B)[0];
    int H    = TensorShape(A)[2];
    int W    = TensorShape(A)[3];
    int fH   = ctx->fH,  fW = ctx->fW;
    int stride = ctx->stride, padding = ctx->padding;
    int Hout = TensorShape(op->out)[2];
    int Wout = TensorShape(op->out)[3];
    int S    = Hout * Wout;
    int K    = Cin * fH * fW;
    int sN = TensorStride(A)[0], sC = TensorStride(A)[1], sH = TensorStride(A)[2], sW = TensorStride(A)[3];
    Matrix Fv = MatrixView(TensorEntries(op->B), K, Cout, 1, K, TensorDataType(op->B), ScaleGet(op->B));
    Real *dProd  = TensorGradients(op->out);
    Real *dInput = TensorGradients(A);

    Real dWin[(size_t)K * S];

    for (int n = start; n < end; n++) {
        Real *dProd_n = dProd + (size_t)n * Cout * S;
        memset(dWin, 0, (size_t)K * S * sizeof(Real));
        MatrixMultiply(Fv,
                       MatrixView(dProd_n, Cout, S, S, 1, FLOAT32, (Scale){0}),
                       MatrixView(dWin, K, S, S, 1, FLOAT32, (Scale){0}));

        for (int ci = 0; ci < Cin; ci++) {
            Real *plane = dInput + (size_t)n * sN + (size_t)ci * sC;
            for (int fh = 0; fh < fH; fh++) {
                for (int fw = 0; fw < fW; fw++) {
                    int k = (ci * fH + fh) * fW + fw;
                    Real *winRow = dWin + (size_t)k * S;
                    for (int oh = 0; oh < Hout; oh++) {
                        int ih = oh * stride - padding + fh;
                        if (ih < 0 || ih >= H) continue;
                        Real *dRow = plane + (size_t)ih * sH;
                        Real *winR = winRow + (size_t)oh * Wout;
                        if (stride == 1 && sW == 1) {
                            int ow0 = padding - fw;     if (ow0 < 0)    ow0 = 0;
                            int ow1 = W + padding - fw; if (ow1 > Wout) ow1 = Wout;
                            Real *dst = dRow + (ow0 - padding + fw);
                            for (int ow = ow0; ow < ow1; ow++)
                                dst[ow - ow0] += winR[ow];
                        } else {
                            for (int ow = 0; ow < Wout; ow++) {
                                int iw = ow * stride - padding + fw;
                                if (iw >= 0 && iw < W)
                                    dRow[(size_t)iw * sW] += winR[ow];
                            }
                        }
                    }
                }
            }
        }
    }
}
