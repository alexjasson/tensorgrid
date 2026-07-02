// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdatomic.h>
#include "../Operation.h"
#include "../math/Matrix.h"

typedef struct {
    int fH, fW, stride, padding;
} ConvolutionContext;

static void OperationConvolutionForward(Operation *op);
static void OperationConvolutionBackward(Operation *op);
static void convolutionForward(void *arg, int start, int end);
static void convolutionBackwardDb(void *arg, int start, int end);
static void convolutionBackwardDa(void *arg, int start, int end);

static void im2col(Operation *op, Entry *patches, int n, int kStart, int kEnd);

Tensor OperationConvolution(Tensor in, Tensor filter, int stride, int padding)
{
    int N    = in.shape[0];
    int H    = in.shape[2];
    int W    = in.shape[3];
    int Cout = filter.shape[0];
    int fH   = filter.shape[2];
    int fW   = filter.shape[3];
    int Hout = (H + 2 * padding - fH) / stride + 1;
    int Wout = (W + 2 * padding - fW) / stride + 1;
    int autograd = in.gradients != NULL || filter.gradients != NULL;

    ConvolutionContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->stride = stride;
    ctx->padding = padding;
    ctx->fH = fH;
    ctx->fW = fW;
    Tensor out = TensorNew(4, (int[]){N, Cout, Hout, Wout}, autograd);
    return OperationExecute(OperationNew(out, in, filter,
                            OperationConvolutionForward, OperationConvolutionBackward, ctx));
}

static void OperationConvolutionForward(Operation *op)
{
    ConvolutionContext *ctx = op->context;
    int Cout = op->B.shape[0];
    int K    = op->B.shape[1] * ctx->fH * ctx->fW;
    int S    = op->out.shape[2] * op->out.shape[3];
    int N    = op->out.shape[0];
    OperationDispatch(convolutionForward, op, N, (long)N * Cout * S * K);
}

static void OperationConvolutionBackward(Operation *op)
{
    ConvolutionContext *ctx = op->context;
    int Cout = op->B.shape[0];
    int K    = op->B.shape[1] * ctx->fH * ctx->fW;
    int S    = op->out.shape[2] * op->out.shape[3];
    int N    = op->out.shape[0];
    long matmulWork = (long)N * Cout * S * K;

    if (op->B.gradients)
        OperationDispatch(convolutionBackwardDb, op, N, matmulWork);
    if (op->A.gradients)
        OperationDispatch(convolutionBackwardDa, op, N, matmulWork);
}

static void im2col(Operation *op, Entry *patches, int n, int kStart, int kEnd)
{
    ConvolutionContext *ctx = op->context;
    Tensor A   = op->A;
    int fH     = ctx->fH,     fW      = ctx->fW;
    int H      = A.shape[2], W       = A.shape[3];
    int Hout   = op->out.shape[2], Wout = op->out.shape[3];
    int stride = ctx->stride, padding = ctx->padding;
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Entry *input = A.entries;
    int S = Hout * Wout;
    for (int k = kStart; k < kEnd; k++) {
        int ci = k / (fH * fW);
        int fh = (k / fW) % fH;
        int fw = k % fW;
        Entry *row     = patches + (size_t)(k - kStart) * S;
        Entry *inPlane = input + (size_t)n * sN + (size_t)ci * sC;
        for (int oh = 0; oh < Hout; oh++) {
            Entry *outRow = row + (size_t)oh * Wout;
            int ih = oh * stride - padding + fh;
            if (ih < 0 || ih >= H) {
                memset(outRow, 0, (size_t)Wout * sizeof(Entry));
                continue;
            }
            Entry *inRow = inPlane + (size_t)ih * sH;
            if (stride == 1 && sW == 1) {
                int ow0 = padding - fw;     if (ow0 < 0)    ow0 = 0;
                int ow1 = W + padding - fw; if (ow1 > Wout) ow1 = Wout;
                if (ow0 > 0)    memset(outRow, 0, (size_t)ow0 * sizeof(Entry));
                if (ow1 > ow0)  memcpy(outRow + ow0, inRow + (ow0 - padding + fw),
                                       (size_t)(ow1 - ow0) * sizeof(Entry));
                if (ow1 < Wout) memset(outRow + ow1, 0, (size_t)(Wout - ow1) * sizeof(Entry));
            } else {
                for (int ow = 0; ow < Wout; ow++) {
                    int iw = ow * stride - padding + fw;
                    outRow[ow] = (iw >= 0 && iw < W) ? inRow[(size_t)iw * sW] : 0.0f;
                }
            }
        }
    }
}

static void convolutionForward(void *arg, int start, int end)
{
    Operation *op = arg;
    ConvolutionContext *ctx = op->context;
    int Cin    = op->B.shape[1];
    int Cout   = op->B.shape[0];
    int fH     = ctx->fH, fW = ctx->fW;
    int S      = op->out.shape[2] * op->out.shape[3];
    int K      = Cin * fH * fW;
    Entry *filter = op->B.entries;
    Entry patches[(size_t)K * S];

    for (int n = start; n < end; n++) {
        im2col(op, patches, n, 0, K);
        Entry *prod = op->out.entries + (size_t)n * Cout * S;
        memset(prod, 0, (size_t)Cout * S * sizeof(Entry));
        MatrixMultiply(MatrixView(filter, Cout, K, K, 1),
                       MatrixView(patches, K, S, S, 1),
                       MatrixView(prod, Cout, S, S, 1));
    }
}

static void convolutionBackwardDb(void *arg, int start, int end)
{
    Operation *op = arg;
    ConvolutionContext *ctx = op->context;
    int Cout   = op->B.shape[0];
    int S      = op->out.shape[2] * op->out.shape[3];
    int K      = op->B.shape[1] * ctx->fH * ctx->fW;
    Entry *dProd   = op->out.gradients;
    Entry *dFilter = op->B.gradients;
    Entry patches[(size_t)K * S];
    Entry local[(size_t)Cout * K];
    memset(local, 0, (size_t)Cout * K * sizeof(Entry));

    for (int n = start; n < end; n++) {
        im2col(op, patches, n, 0, K);
        Entry *dProd_n = dProd + (size_t)n * Cout * S;
        MatrixMultiply(MatrixView(dProd_n, Cout, S, S, 1),
                       MatrixView(patches, S, K, 1, S),
                       MatrixView(local, Cout, K, K, 1));
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
    int Cin  = op->B.shape[1];
    int Cout = op->B.shape[0];
    int H    = A.shape[2];
    int W    = A.shape[3];
    int fH   = ctx->fH,  fW = ctx->fW;
    int stride = ctx->stride, padding = ctx->padding;
    int Hout = op->out.shape[2];
    int Wout = op->out.shape[3];
    int S    = Hout * Wout;
    int K    = Cin * fH * fW;
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Entry *filter = op->B.entries;
    Entry *dProd  = op->out.gradients;
    Entry *dInput = A.gradients;

    Entry dWin[(size_t)K * S];

    for (int n = start; n < end; n++) {
        Entry *dProd_n = dProd + (size_t)n * Cout * S;
        memset(dWin, 0, (size_t)K * S * sizeof(Entry));
        MatrixMultiply(MatrixView(filter, K, Cout, 1, K),
                       MatrixView(dProd_n, Cout, S, S, 1),
                       MatrixView(dWin, K, S, S, 1));

        for (int ci = 0; ci < Cin; ci++) {
            Entry *plane = dInput + (size_t)n * sN + (size_t)ci * sC;
            for (int fh = 0; fh < fH; fh++) {
                for (int fw = 0; fw < fW; fw++) {
                    int k = (ci * fH + fh) * fW + fw;
                    Entry *winRow = dWin + (size_t)k * S;
                    for (int oh = 0; oh < Hout; oh++) {
                        int ih = oh * stride - padding + fh;
                        if (ih < 0 || ih >= H) continue;
                        Entry *dRow = plane + (size_t)ih * sH;
                        Entry *winR = winRow + (size_t)oh * Wout;
                        if (stride == 1 && sW == 1) {
                            int ow0 = padding - fw;     if (ow0 < 0)    ow0 = 0;
                            int ow1 = W + padding - fw; if (ow1 > Wout) ow1 = Wout;
                            Entry *dst = dRow + (ow0 - padding + fw);
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
