// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../utility.h"
#include "../math/Scalar.h"

typedef struct {
    Binary b;
    Real fill;
    int fH, stride, padding;
} PoolContext;

static void OperationPoolForward(Operation *op);
static void OperationPoolBackward(Operation *op);
static void poolForward(void *arg, int start, int end);
static void poolBackward(void *arg, int start, int end);

Tensor OperationPool(Binary fn, Real initial, Tensor in, int size, int stride, int padding)
{
    int N    = in.shape[0];
    int C    = in.shape[1];
    int H    = in.shape[2];
    int W    = in.shape[3];
    int Hout = (H + 2 * padding - size) / stride + 1;
    int Wout = (W + 2 * padding - size) / stride + 1;
    int autograd = in.gradients != NULL;
    int cacheArg = autograd && (fn.f == MAX.f || fn.f == MIN.f);

    Tensor out     = TensorNew(4, (int[]){N, C, Hout, Wout}, REAL, autograd);
    Tensor indices = cacheArg ? TensorNew(4, (int[]){N, C, Hout, Wout}, REAL, FALSE) : (Tensor){0};

    PoolContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    ctx->b = fn;
    ctx->fill = initial;
    ctx->fH = size;
    ctx->stride = stride;
    ctx->padding = padding;

    Tensor result = OperationExecute(OperationNew(out, in, indices,
                    OperationPoolForward, OperationPoolBackward, ctx));
    if (indices.refcount) TensorFree(indices);
    return result;
}

static void OperationPoolForward(Operation *op)
{
    PoolContext *ctx = op->context;
    OperationDispatch(poolForward, op, op->out.shape[0] * op->out.shape[1], (long)TensorSize(op->out) * ctx->fH * ctx->fH);
}

static void OperationPoolBackward(Operation *op)
{
    if (!op->A.gradients) return;
    PoolContext *ctx = op->context;
    int planes = op->A.shape[0] * op->A.shape[1];
    OperationDispatch(poolBackward, op, planes, (long)TensorSize(op->A) * ctx->fH * ctx->fH);
}


static void poolForward(void *arg, int start, int end)
{
    Operation *op = arg;
    PoolContext *ctx = op->context;
    Tensor A = op->A;
    int C    = A.shape[1];
    int H    = A.shape[2];
    int W    = A.shape[3];
    int size = ctx->fH;
    int stride = ctx->stride, padding = ctx->padding;
    int Hout = op->out.shape[2];
    int Wout = op->out.shape[3];
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Real *out    = op->out.entries;
    Real *argIdx = op->B.entries;
    Real fill    = ctx->fill;
    f reducer = ctx->b.f;

    for (int p = start; p < end; p++) {
        int c = p % C, n = p / C;
        int planeOff = n * sN + c * sC;
        size_t outBase = (size_t)p * Hout * Wout;
        for (int oh = 0; oh < Hout; oh++)
            for (int ow = 0; ow < Wout; ow++) {
                size_t outIdx = outBase + (size_t)oh * Wout + ow;
                Real acc = fill;
                int   bl  = -1;
                for (int fh = 0; fh < size; fh++) {
                    int ih = oh * stride - padding + fh;
                    if (ih < 0 || ih >= H) continue;
                    int rowOff = planeOff + ih * sH;
                    for (int fw = 0; fw < size; fw++) {
                        int iw = ow * stride - padding + fw;
                        if (iw < 0 || iw >= W) continue;
                        Real v = TensorGet(A, rowOff + iw * sW);
                        int local = fh * size + fw;
                        if      (reducer == MAX.f) { if (v > acc) { acc = v; bl = local; } }
                        else if (reducer == MIN.f) { if (v < acc) { acc = v; bl = local; } }
                        else if (reducer == ADD.f) { acc += v; }
                        else                       { acc = ((ScalarBinary)reducer)(acc, v); }
                    }
                }
                out[outIdx] = acc;
                if ((reducer == MAX.f || reducer == MIN.f) && argIdx)
                    argIdx[outIdx] = (Real)bl;
            }
    }
}

static void poolBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    PoolContext *ctx = op->context;
    Tensor A      = op->A;
    int C    = A.shape[1];
    int H    = A.shape[2];
    int W    = A.shape[3];
    int size = ctx->fH;
    int stride = ctx->stride, padding = ctx->padding;
    int Hout = op->out.shape[2];
    int Wout = op->out.shape[3];
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Real *input  = A.entries;
    Real *dInput = A.gradients;
    Real *dOut   = op->out.gradients;
    Real *argIdx = op->B.entries;
    Real fill    = ctx->fill;
    f reducer = ctx->b.f;
    ScalarBinary da = (ScalarBinary)ctx->b.da, db = (ScalarBinary)ctx->b.db;

    // Fast paths for max, min and sum
    if (reducer == MAX.f || reducer == MIN.f) {
        if (stride >= size) {
            for (int p = start; p < end; p++) {
                int c = p % C, n = p / C;
                Real *plane   = dInput + (size_t)n * sN + (size_t)c * sC;
                size_t outBase = (size_t)p * Hout * Wout;
                for (int oh = 0; oh < Hout; oh++)
                    for (int ow = 0; ow < Wout; ow++) {
                        size_t outIdx = outBase + (size_t)oh * Wout + ow;
                        int local = (int)argIdx[outIdx];
                        if (local < 0) continue;
                        int ih = oh * stride - padding + local / size;
                        int iw = ow * stride - padding + local % size;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            plane[(size_t)ih * sH + (size_t)iw * sW] += dOut[outIdx];
                    }
            }
        } else {
            for (int p = start; p < end; p++) {
                int c = p % C, n = p / C;
                Real *plane   = dInput + (size_t)n * sN + (size_t)c * sC;
                size_t outBase = (size_t)p * Hout * Wout;
                for (int ih = 0; ih < H; ih++)
                    for (int iw = 0; iw < W; iw++) {
                        int ohMinRaw = ih + padding - size + 1;
                        int ohStart  = ohMinRaw <= 0 ? 0 : (ohMinRaw + stride - 1) / stride;
                        int ohEnd    = (ih + padding) / stride;
                        int owMinRaw = iw + padding - size + 1;
                        int owStart  = owMinRaw <= 0 ? 0 : (owMinRaw + stride - 1) / stride;
                        int owEnd    = (iw + padding) / stride;
                        if (ohEnd >= Hout) ohEnd = Hout - 1;
                        if (owEnd >= Wout) owEnd = Wout - 1;

                        Real grad = 0.0f;
                        for (int oh = ohStart; oh <= ohEnd; oh++)
                            for (int ow = owStart; ow <= owEnd; ow++) {
                                int fh = ih + padding - oh * stride;
                                int fw = iw + padding - ow * stride;
                                size_t outOff = outBase + (size_t)oh * Wout + ow;
                                if ((int)argIdx[outOff] == fh * size + fw)
                                    grad += dOut[outOff];
                            }
                        plane[(size_t)ih * sH + (size_t)iw * sW] += grad;
                    }
            }
        }
    } else if (reducer == ADD.f) {
        for (int p = start; p < end; p++) {
            int c = p % C, n = p / C;
            Real *plane   = dInput + (size_t)n * sN + (size_t)c * sC;
            size_t outBase = (size_t)p * Hout * Wout;
            for (int oh = 0; oh < Hout; oh++)
                for (int ow = 0; ow < Wout; ow++) {
                    Real g = dOut[outBase + (size_t)oh * Wout + ow];
                    for (int fh = 0; fh < size; fh++) {
                        int ih = oh * stride - padding + fh;
                        if (ih < 0 || ih >= H) continue;
                        Real *dr = plane + (size_t)ih * sH;
                        for (int fw = 0; fw < size; fw++) {
                            int iw = ow * stride - padding + fw;
                            if (iw >= 0 && iw < W) dr[(size_t)iw * sW] += g;
                        }
                    }
                }
        }
    } else {
        Real chain[size * size + 1];
        Real vals[size * size];
        size_t offs[size * size];
        for (int p = start; p < end; p++) {
            int c = p % C, n = p / C;
            Real *inPlane = input  + (size_t)n * sN + (size_t)c * sC;
            Real *dPlane  = dInput + (size_t)n * sN + (size_t)c * sC;
            size_t outBase = (size_t)p * Hout * Wout;
            for (int oh = 0; oh < Hout; oh++)
                for (int ow = 0; ow < Wout; ow++) {
                    chain[0] = fill;
                    int cnt = 0;
                    for (int fh = 0; fh < size; fh++) {
                        int ih = oh * stride - padding + fh;
                        if (ih < 0 || ih >= H) continue;
                        for (int fw = 0; fw < size; fw++) {
                            int iw = ow * stride - padding + fw;
                            if (iw < 0 || iw >= W) continue;
                            size_t off = (size_t)ih * sH + (size_t)iw * sW;
                            Real v = inPlane[off];
                            chain[cnt + 1] = ((ScalarBinary)reducer)(chain[cnt], v);
                            vals[cnt] = v;
                            offs[cnt] = off;
                            cnt++;
                        }
                    }
                    Real dres = dOut[outBase + (size_t)oh * Wout + ow];
                    for (int j = cnt - 1; j >= 0; j--) {
                        dPlane[offs[j]] += dres * db(chain[j], vals[j]);
                        dres *= da(chain[j], vals[j]);
                    }
                }
        }
    }
}
