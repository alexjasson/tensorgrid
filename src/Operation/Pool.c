// https://github.com/alexjasson/tensorgrid

#include <stdlib.h>
#include <stdio.h>
#include "../Operation.h"
#include "../math/Vector.h"
#include "../utility.h"

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
    int N    = TensorShape(in)[0];
    int C    = TensorShape(in)[1];
    int H    = TensorShape(in)[2];
    int W    = TensorShape(in)[3];
    int Hout = (H + 2 * padding - size) / stride + 1;
    int Wout = (W + 2 * padding - size) / stride + 1;
    int autograd = TensorGradients(in) != NULL;
    int cacheArg = autograd && (fn.g == MAX.g || fn.g == MIN.g);

    Tensor out     = TensorNew(4, (int[]){N, C, Hout, Wout}, FLOAT32, autograd);
    Tensor indices = cacheArg ? TensorNew(4, (int[]){N, C, Hout, Wout}, FLOAT32, FALSE) : NULL;

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
    if (indices) TensorFree(indices);
    return result;
}

static void OperationPoolForward(Operation *op)
{
    PoolContext *ctx = op->context;
    OperationDispatch(poolForward, op, TensorShape(op->out)[0] * TensorShape(op->out)[1], (long)TensorSize(op->out) * ctx->fH * ctx->fH);
}

static void OperationPoolBackward(Operation *op)
{
    if (!TensorGradients(op->A)) return;
    PoolContext *ctx = op->context;
    int planes = TensorShape(op->A)[0] * TensorShape(op->A)[1];
    OperationDispatch(poolBackward, op, planes, (long)TensorSize(op->A) * ctx->fH * ctx->fH);
}


static void poolForward(void *arg, int start, int end)
{
    Operation *op = arg;
    PoolContext *ctx = op->context;
    Tensor A = op->A;
    int C    = TensorShape(A)[1];
    int H    = TensorShape(A)[2];
    int W    = TensorShape(A)[3];
    int size = ctx->fH;
    int stride = ctx->stride, padding = ctx->padding;
    int Hout = TensorShape(op->out)[2];
    int Wout = TensorShape(op->out)[3];
    int sN = TensorStride(A)[0], sC = TensorStride(A)[1], sH = TensorStride(A)[2], sW = TensorStride(A)[3];
    Real *out    = TensorEntries(op->out);
    Real *argIdx = op->B ? TensorEntries(op->B) : NULL;
    Real fill    = ctx->fill;
    g reducer = ctx->b.g;

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
                        if      (reducer == MAX.g) { if (v > acc) { acc = v; bl = local; } }
                        else if (reducer == MIN.g) { if (v < acc) { acc = v; bl = local; } }
                        else if (reducer == ADD.g) { acc += v; }
                        else                       { acc = VectorApply((VectorBinary)reducer, acc, v); }
                    }
                }
                out[outIdx] = acc;
                if ((reducer == MAX.g || reducer == MIN.g) && argIdx)
                    argIdx[outIdx] = (Real)bl;
            }
    }
}

static void poolBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    PoolContext *ctx = op->context;
    Tensor A      = op->A;
    int C    = TensorShape(A)[1];
    int H    = TensorShape(A)[2];
    int W    = TensorShape(A)[3];
    int size = ctx->fH;
    int stride = ctx->stride, padding = ctx->padding;
    int Hout = TensorShape(op->out)[2];
    int Wout = TensorShape(op->out)[3];
    int sN = TensorStride(A)[0], sC = TensorStride(A)[1], sH = TensorStride(A)[2], sW = TensorStride(A)[3];
    Real *input  = TensorEntries(A);
    Real *dInput = TensorGradients(A);
    Real *dOut   = TensorGradients(op->out);
    Real *argIdx = TensorEntries(op->B);
    Real fill    = ctx->fill;
    g reducer = ctx->b.g;
    VectorBinary da = (VectorBinary)ctx->b.da, db = (VectorBinary)ctx->b.db;

    // Fast paths for max, min and sum
    if (reducer == MAX.g || reducer == MIN.g) {
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
    } else if (reducer == ADD.g) {
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
                            chain[cnt + 1] = VectorApply((VectorBinary)reducer, chain[cnt], v);
                            vals[cnt] = v;
                            offs[cnt] = off;
                            cnt++;
                        }
                    }
                    Real dres = dOut[outBase + (size_t)oh * Wout + ow];
                    for (int j = cnt - 1; j >= 0; j--) {
                        dPlane[offs[j]] += dres * VectorApply(db, chain[j], vals[j]);
                        dres *= VectorApply(da, chain[j], vals[j]);
                    }
                }
        }
    }
}
