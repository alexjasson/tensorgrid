// https://github.com/alexjasson/tensorgrid

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "Operation.h"
#include "utility.h"
#include "ThreadPool.h"
#include "Vec.h"

// Microkernel tile sizes. NR tracks the SIMD width, MR keeps the accumulator tile in registers.
#if !defined(SIMD)
#define MR 4
#define NR 8
#elif defined(__AVX512F__)
#define MR 8
#define NR 16
#elif defined(__AVX2__)
#define MR 6
#define NR 16
#endif

#if defined(SIMD)
#define NRV (NR / VEC_DIMS)
#endif

// Cache block sizes for the temporary packed A/B blocks used by GEMM
#define KC 256
#define NC (NR * 16)
#define MC (MR * 16)
#define PARALLEL (1 << 18) // Parallel threshold

// Strided matrix view of tensor data
typedef struct { Entry *data; int sRow, sCol; } Matrix;

// Helper functions
static void packB(Entry *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock);
static void packA(Entry *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock);
static void microKernel(Entry *Ap, Entry *Bp, int kBlock, Matrix C, int mValid, int nValid);
static void gemm(int N, int K, Matrix A, Matrix B, Matrix C, int r0, int r1);
static void dispatch(KernelFn f, Operation *op, int total, long work);

// Forward kernels
static void mapForward(void *arg, int start, int end);
static void zipForward(void *arg, int start, int end);
static void multiplyForward(void *arg, int start, int end);
static void affineForward(void *arg, int start, int end);
static void convolutionForward(void *arg, int start, int end);
static void poolForward(void *arg, int start, int end);
static void unfoldForward(void *arg, int start, int end);
static void gatherForward(void *arg, int start, int end);
static void scatterForward(void *arg, int start, int end);

// Backward kernels
static void mapBackward(void *arg, int start, int end);
static void zipBackward(void *arg, int start, int end);
static void multiplyBackwardDa(void *arg, int start, int end);
static void multiplyBackwardDb(void *arg, int start, int end);
static void affineBackwardDa(void *arg, int start, int end);
static void affineBackwardDb(void *arg, int start, int end);
static void convolutionBackwardDb(void *arg, int start, int end);
static void convolutionBackwardDa(void *arg, int start, int end);
static void poolBackward(void *arg, int start, int end);
static void unfoldBackward(void *arg, int start, int end);
static void gatherBackward(void *arg, int start, int end);
static void scatterBackward(void *arg, int start, int end);

static void mapForward(void *arg, int start, int end)
{
    Operation *op  = arg;
    Entry     *out = op->out.entries;
    Entry     *A   = op->A.entries;
    UnaryScalar f = op->u.f;
    if (isContiguous(op->A.shape, op->A.strides, op->A.dimensions)) {
        for (int i = start; i < end; i++)
            out[i] = f(A[i]);
    } else {
        for (int i = start; i < end; i++) {
            int aOff = unravel(i, op->A.shape, op->A.strides, op->A.dimensions);
            out[i] = f(A[aOff]);
        }
    }
}

static void zipForward(void *arg, int start, int end)
{
    Operation *op  = arg;
    Entry     *out = op->out.entries;
    Entry     *A   = op->A.entries;
    Entry     *B   = op->B.entries;
    BinaryScalar f  = op->b.f;
    int aC = isContiguous(op->A.shape, op->A.strides, op->A.dimensions);
    int bC = isContiguous(op->B.shape, op->B.strides, op->B.dimensions);
    for (int i = start; i < end; i++) {
        int aOff = aC ? i : unravel(i, op->A.shape, op->A.strides, op->A.dimensions);
        int bOff = bC ? i : unravel(i, op->B.shape, op->B.strides, op->B.dimensions);
        out[i] = f(A[aOff], B[bOff]);
    }
}

// Pack a B block into NR-wide panels, padding the right edge with zeros
static void packB(Entry *Bp, Matrix B, int p0, int kBlock, int j0, int nBlock)
{
    int panels = (nBlock + NR - 1) / NR;
    for (int jp = 0; jp < panels; jp++) {
        Entry *dst = Bp + (size_t)jp * kBlock * NR;
        for (int p = 0; p < kBlock; p++) {
            Entry *brow = B.data + (size_t)(p0 + p) * B.sRow;
            for (int nr = 0; nr < NR; nr++) {
                int col = jp * NR + nr;
                dst[p * NR + nr] = (col < nBlock) ? brow[(size_t)(j0 + col) * B.sCol] : 0.0f;
            }
        }
    }
}

// Pack an A block into MR-row panels, padding the bottom edge with zeros
static void packA(Entry *Ap, Matrix A, int i0, int mBlock, int p0, int kBlock)
{
    int panels = (mBlock + MR - 1) / MR;
    for (int ip = 0; ip < panels; ip++) {
        Entry *dst = Ap + (size_t)ip * kBlock * MR;
        for (int p = 0; p < kBlock; p++) {
            for (int mr = 0; mr < MR; mr++) {
                int row = ip * MR + mr;
                dst[p * MR + mr] = (row < mBlock)
                    ? A.data[(size_t)(i0 + row) * A.sRow + (size_t)(p0 + p) * A.sCol] : 0.0f;
            }
        }
    }
}

// Compute one packed MR x NR tile and add the valid region back into C
static void microKernel(Entry *Ap, Entry *Bp, int kBlock, Matrix C, int mValid, int nValid)
{
#if defined(SIMD)
    Vec acc[MR][NRV];
    for (int mr = 0; mr < MR; mr++)
        for (int v = 0; v < NRV; v++)
            acc[mr][v] = VecZero();

    for (int p = 0; p < kBlock; p++) {
        Vec b[NRV];
        for (int v = 0; v < NRV; v++)
            b[v] = VecLoad(Bp + p * NR + v * VEC_DIMS);
        for (int mr = 0; mr < MR; mr++) {
            Vec a = VecBcast(Ap[p * MR + mr]);
            for (int v = 0; v < NRV; v++)
                acc[mr][v] = VecFma(a, b[v], acc[mr][v]);
        }
    }

    _Alignas(64) Entry tile[MR][NR];
    for (int mr = 0; mr < MR; mr++)
        for (int v = 0; v < NRV; v++)
            VecStore(&tile[mr][v * VEC_DIMS], acc[mr][v]);
    for (int mr = 0; mr < mValid; mr++)
        for (int nr = 0; nr < nValid; nr++)
            C.data[mr * C.sRow + nr * C.sCol] += tile[mr][nr];
#else
    Entry acc[MR][NR] = {{0}};
    for (int p = 0; p < kBlock; p++) {
        Entry *a = Ap + p * MR;
        Entry *b = Bp + p * NR;
        for (int mr = 0; mr < MR; mr++)
            for (int nr = 0; nr < NR; nr++)
                acc[mr][nr] += a[mr] * b[nr];
    }
    for (int mr = 0; mr < mValid; mr++)
        for (int nr = 0; nr < nValid; nr++)
            C.data[mr * C.sRow + nr * C.sCol] += acc[mr][nr];
#endif
}

// cache-blocked packed GEMM
static void gemm(int N, int K, Matrix A, Matrix B, Matrix C, int r0, int r1)
{
    _Alignas(64) Entry packBufA[MC * KC];
    _Alignas(64) Entry packBufB[NC * KC];

    for (int j0 = 0; j0 < N; j0 += NC) {
        int nBlock = min(NC, N - j0);
        for (int p0 = 0; p0 < K; p0 += KC) {
            int kBlock = min(KC, K - p0);
            packB(packBufB, B, p0, kBlock, j0, nBlock);
            for (int i0 = r0; i0 < r1; i0 += MC) {
                int mBlock = min(MC, r1 - i0);
                packA(packBufA, A, i0, mBlock, p0, kBlock);
                for (int jr = 0; jr < nBlock; jr += NR) {
                    Entry *Bpanel = packBufB + (size_t)(jr / NR) * kBlock * NR;
                    for (int ir = 0; ir < mBlock; ir += MR) {
                        Entry *Apanel = packBufA + (size_t)(ir / MR) * kBlock * MR;
                        Matrix Ctile = { C.data + (size_t)(i0 + ir) * C.sRow + (size_t)(j0 + jr) * C.sCol, C.sRow, C.sCol };
                        microKernel(Apanel, Bpanel, kBlock, Ctile, min(MR, mBlock - ir), min(NR, nBlock - jr));
                    }
                }
            }
        }
    }
}

// out = A @ B
static void multiplyForward(void *arg, int start, int end)
{
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int strideA_row  = op->A.strides[op->A.dimensions - 2];
    int strideA_col  = op->A.strides[op->A.dimensions - 1];
    int strideB_row  = op->B.strides[op->B.dimensions - 2];
    int strideB_col  = op->B.strides[op->B.dimensions - 1];
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;

    int i = start;
    while (i < end) {
        int batch      = i / m;
        int r0         = i % m;               // first row of this batch handled here
        int batchEnd   = (batch + 1) * m;
        int chunkEnd   = (end < batchEnd) ? end : batchEnd;
        int r1         = r0 + (chunkEnd - i); // one past the last row
        Entry *A_batch = op->A.entries   + (size_t)batch * batchStrideA;
        Entry *B_batch = op->B.entries   + (size_t)batch * batchStrideB;
        Entry *C_batch = op->out.entries + (size_t)batch * m * n;

        memset(C_batch + (size_t)r0 * n, 0, (size_t)(r1 - r0) * n * sizeof(Entry));
        gemm(n, k, (Matrix){A_batch, strideA_row, strideA_col},
                   (Matrix){B_batch, strideB_row, strideB_col},
                   (Matrix){C_batch, n, 1}, r0, r1);
        i = chunkEnd;
    }
}

static void mapBackward(void *arg, int start, int end)
{
    Operation   *op = arg;
    Entry       *A  = op->A.entries;
    Entry       *dA = op->A.gradients;
    Entry     *dout = op->out.gradients;
    UnaryScalar da = op->u.da;
    if (isContiguous(op->A.shape, op->A.strides, op->A.dimensions)) {
        for (int j = start; j < end; j++)
            dA[j] += dout[j] * da(A[j]);
    } else {
        for (int j = start; j < end; j++) {
            int aOff = unravel(j, op->A.shape, op->A.strides, op->A.dimensions);
            dA[aOff] += dout[j] * da(A[aOff]);
        }
    }
}

static void zipBackward(void *arg, int start, int end)
{
    Operation  *op = arg;
    Entry       *A = op->A.entries;
    Entry       *B = op->B.entries;
    Entry      *dA = op->A.gradients;
    Entry      *dB = op->B.gradients;
    Entry    *dout = op->out.gradients;
    BinaryScalar da = op->b.da;
    BinaryScalar db = op->b.db;
    int aC = isContiguous(op->A.shape, op->A.strides, op->A.dimensions);
    int bC = isContiguous(op->B.shape, op->B.strides, op->B.dimensions);
    for (int j = start; j < end; j++) {
        int aOff = aC ? j : unravel(j, op->A.shape, op->A.strides, op->A.dimensions);
        int bOff = bC ? j : unravel(j, op->B.shape, op->B.strides, op->B.dimensions);
        Entry g = dout[j];
        if (dA) dA[aOff] += g * da(A[aOff], B[bOff]);
        if (dB) dB[bOff] += g * db(A[aOff], B[bOff]);
    }
}

// out = f(A + bias)
static void affineForward(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *out = op->out.entries, *A = op->A.entries, *bias = op->B.entries;
    int biasSize = TensorSize(op->B), inner = op->inner;
    UnaryScalar f = op->u.f;
#if defined(SIMD)
    Vec (*fv)(Vec) = (Vec (*)(Vec))op->u.fv;
#endif
    if (inner == 1) {
        for (int o = start; o < end; o++) {
            size_t base = (size_t)o * biasSize;
            int c = 0;
#if defined(SIMD)
            if (fv) for (; c + VEC_DIMS <= biasSize; c += VEC_DIMS)
                VecStore(out + base + c, fv(VecAdd(VecLoad(A + base + c), VecLoad(bias + c))));
#endif
            for (; c < biasSize; c++) out[base + c] = f(A[base + c] + bias[c]);
        }
    } else {
        for (int o = start; o < end; o++)
            for (int c = 0; c < biasSize; c++) {
                Entry bc = bias[c];
                size_t base = ((size_t)o * biasSize + c) * inner;
                int i = 0;
#if defined(SIMD)
                if (fv) {
                    Vec bcv = VecBcast(bc);
                    for (; i + VEC_DIMS <= inner; i += VEC_DIMS)
                        VecStore(out + base + i, fv(VecAdd(VecLoad(A + base + i), bcv)));
                }
#endif
                for (; i < inner; i++) out[base + i] = f(A[base + i] + bc);
            }
    }
}

// dA += dout * f'(A + bias)
static void affineBackwardDa(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *A = op->A.entries, *bias = op->B.entries, *dA = op->A.gradients, *dout = op->out.gradients;
    int biasSize = TensorSize(op->B), inner = op->inner;
    UnaryScalar da = op->u.da;
#if defined(SIMD)
    Vec (*dav)(Vec) = (Vec (*)(Vec))op->u.dav;
#endif
    if (inner == 1) {
        for (int o = start; o < end; o++) {
            size_t base = (size_t)o * biasSize;
            int c = 0;
#if defined(SIMD)
            if (dav) for (; c + VEC_DIMS <= biasSize; c += VEC_DIMS) {
                Vec v = VecAdd(VecLoad(A + base + c), VecLoad(bias + c));
                VecStore(dA + base + c, VecFma(VecLoad(dout + base + c), dav(v), VecLoad(dA + base + c)));
            }
#endif
            for (; c < biasSize; c++) { Entry v = A[base + c] + bias[c]; dA[base + c] += dout[base + c] * da(v); }
        }
    } else {
        for (int o = start; o < end; o++)
            for (int c = 0; c < biasSize; c++) {
                Entry bc = bias[c];
                size_t base = ((size_t)o * biasSize + c) * inner;
                int i = 0;
#if defined(SIMD)
                if (dav) {
                    Vec bcv = VecBcast(bc);
                    for (; i + VEC_DIMS <= inner; i += VEC_DIMS) {
                        Vec v = VecAdd(VecLoad(A + base + i), bcv);
                        VecStore(dA + base + i, VecFma(VecLoad(dout + base + i), dav(v), VecLoad(dA + base + i)));
                    }
                }
#endif
                for (; i < inner; i++) { Entry v = A[base + i] + bc; dA[base + i] += dout[base + i] * da(v); }
            }
    }
}

// dB[c] += sum(dout * f'(A + bias[c]))
static void affineBackwardDb(void *arg, int start, int end)
{
    Operation *op = arg;
    Entry *A = op->A.entries, *bias = op->B.entries, *dB = op->B.gradients, *dout = op->out.gradients;
    int biasSize = TensorSize(op->B), inner = op->inner;
    int outer = TensorSize(op->out) / (biasSize * inner);
    UnaryScalar da = op->u.da;
#if defined(SIMD)
    Vec (*dav)(Vec) = (Vec (*)(Vec))op->u.dav;
#endif
    for (int c = start; c < end; c++) {
        Entry s = 0, bc = bias[c];
#if defined(SIMD)
        Vec sv = VecZero();
#endif
        for (int o = 0; o < outer; o++) {
            size_t base = ((size_t)o * biasSize + c) * inner;
            int i = 0;
#if defined(SIMD)
            if (dav) {
                Vec bcv = VecBcast(bc);
                for (; i + VEC_DIMS <= inner; i += VEC_DIMS) {
                    Vec v = VecAdd(VecLoad(A + base + i), bcv);
                    sv = VecFma(VecLoad(dout + base + i), dav(v), sv);
                }
            }
#endif
            for (; i < inner; i++) { Entry v = A[base + i] + bc; s += dout[base + i] * da(v); }
        }
#if defined(SIMD)
        s += VecHsum(sv);
#endif
        dB[c] += s;
    }
}

// dA += dout @ B^T.
static void multiplyBackwardDa(void *arg, int start, int end)
{
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int batchSize    = TensorSize(op->out) / (m * n);
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;
    int aRowStride   = op->A.strides[op->A.dimensions - 2];
    int aColStride   = op->A.strides[op->A.dimensions - 1];
    int bRowStride   = op->B.strides[op->B.dimensions - 2];
    int bColStride   = op->B.strides[op->B.dimensions - 1];

    if (batchStrideA) {
        int i = start;
        while (i < end) {
            int batch    = i / m;
            int r0       = i % m;
            int batchEnd = (batch + 1) * m;
            int chunkEnd = (end < batchEnd) ? end : batchEnd;
            int r1       = r0 + (chunkEnd - i);
            Entry *dout  = op->out.gradients + (size_t)batch * m * n;
            Entry *B     = op->B.entries     + (size_t)batch * batchStrideB;
            Entry *dA    = op->A.gradients   + (size_t)batch * batchStrideA;
            gemm(k, n, (Matrix){dout, n, 1}, (Matrix){B, bColStride, bRowStride},
                       (Matrix){dA, aRowStride, aColStride}, r0, r1);
            i = chunkEnd;
        }
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            Entry *B    = op->B.entries     + (size_t)batch * batchStrideB;
            gemm(m, n, (Matrix){B, bRowStride, bColStride}, (Matrix){dout, 1, n},
                       (Matrix){op->A.gradients, aColStride, aRowStride}, start, end);
        }
    }
}

// dB += A^T @ dout.
static void multiplyBackwardDb(void *arg, int start, int end)
{
    Operation *op    = arg;
    int m            = op->A.shape[op->A.dimensions - 2];
    int k            = op->A.shape[op->A.dimensions - 1];
    int n            = op->B.shape[op->B.dimensions - 1];
    int batchSize    = TensorSize(op->out) / (m * n);
    int batchStrideA = (op->A.dimensions >= 3) ? op->A.strides[op->A.dimensions - 3] : 0;
    int batchStrideB = (op->B.dimensions >= 3) ? op->B.strides[op->B.dimensions - 3] : 0;
    int aRowStride   = op->A.strides[op->A.dimensions - 2];
    int aColStride   = op->A.strides[op->A.dimensions - 1];
    int bRowStride   = op->B.strides[op->B.dimensions - 2];
    int bColStride   = op->B.strides[op->B.dimensions - 1];

    if (batchStrideB) {
        int i = start;
        while (i < end) {
            int batch    = i / k;
            int r0       = i % k;
            int batchEnd = (batch + 1) * k;
            int chunkEnd = (end < batchEnd) ? end : batchEnd;
            int r1       = r0 + (chunkEnd - i);
            Entry *A     = op->A.entries     + (size_t)batch * batchStrideA;
            Entry *dout  = op->out.gradients + (size_t)batch * m * n;
            Entry *dB    = op->B.gradients   + (size_t)batch * batchStrideB;
            gemm(n, m, (Matrix){A, aColStride, aRowStride}, (Matrix){dout, n, 1},
                       (Matrix){dB, bRowStride, bColStride}, r0, r1);
            i = chunkEnd;
        }
    } else {
        for (int batch = 0; batch < batchSize; batch++) {
            Entry *A    = op->A.entries     + (size_t)batch * batchStrideA;
            Entry *dout = op->out.gradients + (size_t)batch * m * n;
            gemm(n, m, (Matrix){A, aColStride, aRowStride}, (Matrix){dout, n, 1},
                       (Matrix){op->B.gradients, bRowStride, bColStride}, start, end);
        }
    }
}

static void im2col(Operation *op, Entry *patches, int n)
{
    Tensor A   = op->A;
    int Cin    = op->B.shape[1];
    int fH     = op->fH,     fW      = op->fW;
    int H      = A.shape[2], W       = A.shape[3];
    int Hout   = op->out.shape[2], Wout = op->out.shape[3];
    int stride = op->stride, padding = op->padding;
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Entry *input = A.entries;
    int S = Hout * Wout;
    int K = Cin * fH * fW;
    for (int k = 0; k < K; k++) {
        int ci = k / (fH * fW);
        int fh = (k / fW) % fH;
        int fw = k % fW;
        Entry *row     = patches + (size_t)k * S;
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
    int Cin    = op->B.shape[1];
    int Cout   = op->B.shape[0];
    int fH     = op->fH, fW = op->fW;
    int S      = op->out.shape[2] * op->out.shape[3];
    int K      = Cin * fH * fW;
    Entry *filter = op->B.entries;
    Entry patches[(size_t)K * S];

    for (int n = start; n < end; n++) {
        im2col(op, patches, n);
        Entry *prod = op->product.entries + (size_t)n * Cout * S;
        memset(prod, 0, (size_t)Cout * S * sizeof(Entry));
        gemm(S, K, (Matrix){filter, K, 1}, (Matrix){patches, S, 1},
                   (Matrix){prod, S, 1}, 0, Cout);
    }
}

static void convolutionBackwardDb(void *arg, int start, int end)
{
    Operation *op = arg;
    int Cin    = op->B.shape[1];
    int Cout   = op->B.shape[0];
    int fH     = op->fH, fW = op->fW;
    int S      = op->out.shape[2] * op->out.shape[3];
    int K      = Cin * fH * fW;
    Entry *dProd = op->product.gradients;
    Entry patches[(size_t)K * S];

    for (int n = start; n < end; n++) {
        im2col(op, patches, n);
        Entry *partial = op->scratch.entries + (size_t)n * Cout * K;
        Entry *dProd_n = dProd       + (size_t)n * Cout * S;
        memset(partial, 0, (size_t)Cout * K * sizeof(Entry));
        gemm(K, S, (Matrix){dProd_n, S, 1}, (Matrix){patches, 1, S},
                   (Matrix){partial, K, 1}, 0, Cout);
    }
}

static void convolutionBackwardDa(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A = op->A;
    int Cin  = op->B.shape[1];
    int Cout = op->B.shape[0];
    int H    = A.shape[2];
    int W    = A.shape[3];
    int fH   = op->fH,  fW = op->fW;
    int stride = op->stride, padding = op->padding;
    int Hout = op->out.shape[2];
    int Wout = op->out.shape[3];
    int S    = Hout * Wout;
    int K    = Cin * fH * fW;
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Entry *filter = op->B.entries;
    Entry *dProd  = op->product.gradients;
    Entry *dInput = A.gradients;

    Entry dWin[(size_t)K * S];

    for (int n = start; n < end; n++) {
        Entry *dProd_n = dProd + (size_t)n * Cout * S;
        memset(dWin, 0, (size_t)K * S * sizeof(Entry));
        gemm(S, Cout, (Matrix){filter, 1, K}, (Matrix){dProd_n, S, 1},
                      (Matrix){dWin, S, 1}, 0, K);

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

static void poolForward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A = op->A;
    int C    = A.shape[1];
    int H    = A.shape[2];
    int W    = A.shape[3];
    int size = op->fH;
    int stride = op->stride, padding = op->padding;
    int Hout = op->out.shape[2];
    int Wout = op->out.shape[3];
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Entry *input  = A.entries;
    Entry *out    = op->out.entries;
    Entry *argIdx = op->product.entries;
    Entry fill    = op->fill;
    BinaryScalar f = op->b.f;

    for (int p = start; p < end; p++) {
        int c = p % C, n = p / C;
        Entry *plane   = input + (size_t)n * sN + (size_t)c * sC;
        size_t outBase = (size_t)p * Hout * Wout;
        for (int oh = 0; oh < Hout; oh++)
            for (int ow = 0; ow < Wout; ow++) {
                size_t outIdx = outBase + (size_t)oh * Wout + ow;
                Entry acc = fill;
                int   bl  = -1;
                for (int fh = 0; fh < size; fh++) {
                    int ih = oh * stride - padding + fh;
                    if (ih < 0 || ih >= H) continue;
                    Entry *r = plane + (size_t)ih * sH;
                    for (int fw = 0; fw < size; fw++) {
                        int iw = ow * stride - padding + fw;
                        if (iw < 0 || iw >= W) continue;
                        Entry v = r[(size_t)iw * sW];
                        int local = fh * size + fw;
                        if      (f == MAX.f) { if (v > acc) { acc = v; bl = local; } }
                        else if (f == MIN.f) { if (v < acc) { acc = v; bl = local; } }
                        else if (f == ADD.f) { acc += v; }
                        else                 { acc = f(acc, v); }
                    }
                }
                out[outIdx] = acc;
                if ((f == MAX.f || f == MIN.f) && argIdx)
                    argIdx[outIdx] = (Entry)bl;
            }
    }
}

static void poolBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    int C    = A.shape[1];
    int H    = A.shape[2];
    int W    = A.shape[3];
    int size = op->fH;
    int stride = op->stride, padding = op->padding;
    int Hout = op->out.shape[2];
    int Wout = op->out.shape[3];
    int sN = A.strides[0], sC = A.strides[1], sH = A.strides[2], sW = A.strides[3];
    Entry *input  = A.entries;
    Entry *dInput = A.gradients;
    Entry *dOut   = op->out.gradients;
    Entry *argIdx = op->product.entries;
    Entry fill    = op->fill;
    BinaryScalar f = op->b.f, da = op->b.da, db = op->b.db;

    // Fast paths for max, min and sum
    if (f == MAX.f || f == MIN.f) {
        if (stride >= size) {
            for (int p = start; p < end; p++) {
                int c = p % C, n = p / C;
                Entry *plane   = dInput + (size_t)n * sN + (size_t)c * sC;
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
                Entry *plane   = dInput + (size_t)n * sN + (size_t)c * sC;
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

                        Entry grad = 0.0f;
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
    } else if (f == ADD.f) {
        for (int p = start; p < end; p++) {
            int c = p % C, n = p / C;
            Entry *plane   = dInput + (size_t)n * sN + (size_t)c * sC;
            size_t outBase = (size_t)p * Hout * Wout;
            for (int oh = 0; oh < Hout; oh++)
                for (int ow = 0; ow < Wout; ow++) {
                    Entry g = dOut[outBase + (size_t)oh * Wout + ow];
                    for (int fh = 0; fh < size; fh++) {
                        int ih = oh * stride - padding + fh;
                        if (ih < 0 || ih >= H) continue;
                        Entry *dr = plane + (size_t)ih * sH;
                        for (int fw = 0; fw < size; fw++) {
                            int iw = ow * stride - padding + fw;
                            if (iw >= 0 && iw < W) dr[(size_t)iw * sW] += g;
                        }
                    }
                }
        }
    } else {
        Entry chain[size * size + 1];
        Entry vals[size * size];
        size_t offs[size * size];
        for (int p = start; p < end; p++) {
            int c = p % C, n = p / C;
            Entry *inPlane = input  + (size_t)n * sN + (size_t)c * sC;
            Entry *dPlane  = dInput + (size_t)n * sN + (size_t)c * sC;
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
                            Entry v = inPlane[off];
                            chain[cnt + 1] = f(chain[cnt], v);
                            vals[cnt] = v;
                            offs[cnt] = off;
                            cnt++;
                        }
                    }
                    Entry dres = dOut[outBase + (size_t)oh * Wout + ow];
                    for (int j = cnt - 1; j >= 0; j--) {
                        dPlane[offs[j]] += dres * db(chain[j], vals[j]);
                        dres *= da(chain[j], vals[j]);
                    }
                }
        }
    }
}

static void unfoldForward(void *arg, int start, int end)
{
    Operation *op = arg;
    int Cin    = op->A.shape[1];
    int H      = op->A.shape[2];
    int W      = op->A.shape[3];
    int fH     = op->fH,     fW      = op->fW;
    int stride = op->stride, padding = op->padding;
    int Hout   = (H + 2 * padding - fH) / stride + 1;
    int Wout   = (W + 2 * padding - fW) / stride + 1;
    int S      = Hout * Wout;
    int K      = Cin * fH * fW;
    int sN     = op->A.strides[0];
    int sC     = op->A.strides[1];
    int sH     = op->A.strides[2];
    int sW     = op->A.strides[3];
    Entry fill     = op->fill;
    Entry *input   = op->A.entries;
    Entry *windows = op->out.entries;

    for (int i = start; i < end; i++) {
        int n  = i / K;
        int k  = i % K;
        int ci = k / (fH * fW);
        int fh = (k / fW) % fH;
        int fw = k % fW;
        Entry *windowRow = windows + (n * K + k) * S;
        Entry *inPlane   = input + n * sN + ci * sC;

        for (int oh = 0; oh < Hout; oh++) {
            int ih = oh * stride - padding + fh;
            if (ih < 0 || ih >= H) {
                for (int ow = 0; ow < Wout; ow++)
                    windowRow[oh * Wout + ow] = fill;
            } else {
                Entry *row = inPlane + ih * sH;
                for (int ow = 0; ow < Wout; ow++) {
                    int iw = ow * stride - padding + fw;
                    windowRow[oh * Wout + ow] = (iw >= 0 && iw < W) ? row[iw * sW] : fill;
                }
            }
        }
    }
}

static void unfoldBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    int Cin    = op->A.shape[1];
    int H      = op->A.shape[2];
    int W      = op->A.shape[3];
    int fH     = op->fH,     fW      = op->fW;
    int stride = op->stride, padding = op->padding;
    int Hout   = (H + 2 * padding - fH) / stride + 1;
    int Wout   = (W + 2 * padding - fW) / stride + 1;
    int S      = Hout * Wout;
    int K      = Cin * fH * fW;
    int sN     = op->A.strides[0];
    int sC     = op->A.strides[1];
    int sH     = op->A.strides[2];
    int sW     = op->A.strides[3];
    Entry *dInput  = op->A.gradients;
    Entry *dWindow = op->out.gradients;

    for (int i = start; i < end; i++) {
        int n  = i / Cin;
        int ci = i % Cin;
        Entry *plane = dInput + (size_t)n * sN + (size_t)ci * sC;

        for (int fh = 0; fh < fH; fh++) {
            for (int fw = 0; fw < fW; fw++) {
                int k = (ci * fH + fh) * fW + fw;
                Entry *windowRow = dWindow + ((size_t)n * K + k) * S;
                for (int oh = 0; oh < Hout; oh++) {
                    int ih = oh * stride - padding + fh;
                    if (ih < 0 || ih >= H) continue;
                    Entry *dRow   = plane + (size_t)ih * sH;
                    Entry *winRow = windowRow + oh * Wout;
                    for (int ow = 0; ow < Wout; ow++) {
                        int iw = ow * stride - padding + fw;
                        if (iw >= 0 && iw < W)
                            dRow[iw * sW] += winRow[ow];
                    }
                }
            }
        }
    }
}

static void gatherForward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int axisSize  = A.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < A.dimensions; d++) innerSize *= A.shape[d];
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    if (aC && iC) {
        for (int i = start; i < end; i++) {
            int inner = i % innerSize;
            int rem   = i / innerSize;
            int ix    = rem % idxSize;
            int outer = rem / idxSize;
            int k     = (int)I.entries[ix];
            op->out.entries[i] = A.entries[(outer * axisSize + k) * innerSize + inner];
        }
    } else {
        for (int i = start; i < end; i++) {
            int inner = i % innerSize;
            int rem   = i / innerSize;
            int ix    = rem % idxSize;
            int outer = rem / idxSize;

            int outerOff = aC ? outer * axisSize * innerSize
                              : unravel(outer, A.shape, A.strides, axis);
            int innerOff = aC ? inner
                              : unravel(inner, A.shape + axis + 1, A.strides + axis + 1,
                                        A.dimensions - axis - 1);
            int aStride  = aC ? innerSize : A.strides[axis];
            int idxOff   = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
            int k        = (int)I.entries[idxOff];
            op->out.entries[i] = A.entries[outerOff + k * aStride + innerOff];
        }
    }
}

static void gatherBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int axisSize  = A.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < A.dimensions; d++) innerSize *= A.shape[d];
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    for (int i = start; i < end; i++) {
        int inner   = i % innerSize;
        int rem     = i / innerSize;
        int kTarget = rem % axisSize;
        int outer   = rem / axisSize;

        int outerOff, innerOff, aStride;
        if (aC) {
            outerOff = outer * axisSize * innerSize;
            innerOff = inner;
            aStride  = innerSize;
        } else {
            outerOff = unravel(outer, A.shape, A.strides, axis);
            innerOff = unravel(inner, A.shape + axis + 1, A.strides + axis + 1,
                               A.dimensions - axis - 1);
            aStride  = A.strides[axis];
        }

        Entry sum = 0;
        if (iC) {
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)I.entries[ix] == kTarget)
                    sum += op->out.gradients[(outer * idxSize + ix) * innerSize + inner];
            }
        } else {
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = unravel(ix, I.shape, I.strides, I.dimensions);
                if ((int)I.entries[idxOff] == kTarget)
                    sum += op->out.gradients[(outer * idxSize + ix) * innerSize + inner];
            }
        }
        A.gradients[outerOff + kTarget * aStride + innerOff] += sum;
    }
}

static void scatterForward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int n         = op->out.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) innerSize *= op->out.shape[d];
    int aInnerDims = A.dimensions - axis - I.dimensions;
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    if (aC && iC) {
        for (int i = start; i < end; i++) {
            int inner   = i % innerSize;
            int rem     = i / innerSize;
            int kTarget = rem % n;
            int outer   = rem / n;

            Entry acc = op->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                if ((int)I.entries[ix] == kTarget)
                    acc = op->b.f(acc, A.entries[(outer * idxSize + ix) * innerSize + inner]);
            }
            op->out.entries[i] = acc;
        }
    } else {
        for (int i = start; i < end; i++) {
            int inner   = i % innerSize;
            int rem     = i / innerSize;
            int kTarget = rem % n;
            int outer   = rem / n;

            int outerOff = aC ? outer * idxSize * innerSize
                              : unravel(outer, A.shape, A.strides, axis);
            int innerOff = aC ? inner
                              : unravel(inner, A.shape + axis + I.dimensions,
                                        A.strides + axis + I.dimensions, aInnerDims);

            Entry acc = op->fill;
            for (int ix = 0; ix < idxSize; ix++) {
                int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
                if ((int)I.entries[idxOff] == kTarget) {
                    int aIxOff = aC ? ix * innerSize
                                    : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                    acc = op->b.f(acc, A.entries[outerOff + aIxOff + innerOff]);
                }
            }
            op->out.entries[i] = acc;
        }
    }
}

static void scatterBackward(void *arg, int start, int end)
{
    Operation *op = arg;
    Tensor A      = op->A;
    Tensor I      = op->B;
    int axis      = op->axis;
    int n         = op->out.shape[axis];
    int idxSize   = TensorSize(I);
    int innerSize = 1;
    for (int d = axis + 1; d < op->out.dimensions; d++) innerSize *= op->out.shape[d];
    int aInnerDims = A.dimensions - axis - I.dimensions;
    int aC = isContiguous(A.shape, A.strides, A.dimensions);
    int iC = isContiguous(I.shape, I.strides, I.dimensions);

    for (int i = start; i < end; i++) {
        int inner   = i % innerSize;
        int rem     = i / innerSize;
        int kTarget = rem % n;
        int outer   = rem / n;

        int outerOff, innerOff;
        if (aC) {
            outerOff = outer * idxSize * innerSize;
            innerOff = inner;
        } else {
            outerOff = unravel(outer, A.shape, A.strides, axis);
            innerOff = unravel(inner, A.shape + axis + I.dimensions,
                               A.strides + axis + I.dimensions, aInnerDims);
        }

        Entry chain[idxSize + 1];
        chain[0] = op->fill;
        int count = 0;
        for (int ix = 0; ix < idxSize; ix++) {
            int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
            if ((int)I.entries[idxOff] == kTarget) {
                int aIxOff = aC ? ix * innerSize
                                : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                Entry Av = A.entries[outerOff + aIxOff + innerOff];
                chain[count + 1] = op->b.f(chain[count], Av);
                count++;
            }
        }

        Entry dresult = op->out.gradients[i];
        int c = count - 1;
        for (int ix = idxSize - 1; ix >= 0 && c >= 0; ix--) {
            int idxOff = iC ? ix : unravel(ix, I.shape, I.strides, I.dimensions);
            if ((int)I.entries[idxOff] == kTarget) {
                int aIxOff  = aC ? ix * innerSize
                                 : unravel(ix, A.shape + axis, A.strides + axis, I.dimensions);
                int aOffset = outerOff + aIxOff + innerOff;
                Entry Av    = A.entries[aOffset];
                A.gradients[aOffset] += dresult * op->b.db(chain[c], Av);
                dresult *= op->b.da(chain[c], Av);
                c--;
            }
        }
    }
}

// If work is more than parallel threshold, parallelize with thread pool
static void dispatch(KernelFn f, Operation *op, int total, long work)
{
    if (total <= 0) return;
    if (work < PARALLEL) f(op, 0, total);
    else ThreadPoolMap(ThreadPoolGet(), f, op, total);
}

void OperationForward(Operation op)
{
    switch (op.type) {
        case Map:
            dispatch(mapForward, &op, TensorSize(op.out), TensorSize(op.out));
            break;
        case Zip:
            dispatch(zipForward, &op, TensorSize(op.out), TensorSize(op.out));
            break;
        case Multiply: {
            int m         = op.A.shape[op.A.dimensions - 2];
            int k         = op.A.shape[op.A.dimensions - 1];
            int n         = op.B.shape[op.B.dimensions - 1];
            int batchSize = TensorSize(op.out) / (m * n);
            dispatch(multiplyForward, &op, batchSize * m, (long)TensorSize(op.out) * k);
            break;
        }
        case Affine: {
            int outer = TensorSize(op.out) / (TensorSize(op.B) * op.inner);
            dispatch(affineForward, &op, outer, TensorSize(op.out));
            break;
        }
        case Linear: {
            // out = f(A @ B + bias): matmul into the cached product, then affine
            Operation mul = { .type=Multiply, .out=op.product, .A=op.A, .B=op.B };
            OperationForward(mul);
            Operation aff = { .type=Affine, .out=op.out, .A=op.product, .B=op.bias,
                              .u=op.u, .inner=op.inner };
            OperationForward(aff);
            break;
        }
        case Convolution: {
            int Cout = op.B.shape[0];
            int K    = op.B.shape[1] * op.fH * op.fW;
            int S    = op.out.shape[2] * op.out.shape[3];
            int N    = op.out.shape[0];
            dispatch(convolutionForward, &op, N, (long)N * Cout * S * K);
            Operation aff = { .type=Affine, .out=op.out, .A=op.product, .B=op.bias,
                              .u=op.u, .inner=S };
            OperationForward(aff);
            break;
        }
        case Pool:
            dispatch(poolForward, &op, op.out.shape[0] * op.out.shape[1],
                     (long)TensorSize(op.out) * op.fH * op.fW);
            break;
        case Unfold:
            dispatch(unfoldForward, &op, op.out.shape[0] * op.out.shape[1], TensorSize(op.out));
            break;
        case Gather:
            dispatch(gatherForward, &op, TensorSize(op.out), TensorSize(op.out));
            break;
        case Scatter:
            dispatch(scatterForward, &op, TensorSize(op.out),
                     (long)TensorSize(op.out) * TensorSize(op.B));
            break;
    }
}

void OperationBackward(Operation op)
{
    switch (op.type) {
        case Map:
            if (op.A.gradients)
                dispatch(mapBackward, &op, TensorSize(op.out), TensorSize(op.out));
            break;
        case Zip:
            if (op.A.gradients || op.B.gradients)
                dispatch(zipBackward, &op, TensorSize(op.out), TensorSize(op.out));
            break;
        case Multiply: {
            int m            = op.A.shape[op.A.dimensions - 2];
            int k            = op.A.shape[op.A.dimensions - 1];
            int n            = op.B.shape[op.B.dimensions - 1];
            int batchSize    = TensorSize(op.out) / (m * n);
            int batchStrideA = (op.A.dimensions >= 3) ? op.A.strides[op.A.dimensions - 3] : 0;
            int batchStrideB = (op.B.dimensions >= 3) ? op.B.strides[op.B.dimensions - 3] : 0;
            long work        = (long)TensorSize(op.out) * k;
            if (op.A.gradients)
                dispatch(multiplyBackwardDa, &op, batchStrideA ? batchSize * m : k, work);
            if (op.B.gradients)
                dispatch(multiplyBackwardDb, &op, batchStrideB ? batchSize * k : k, work);
            break;
        }
        case Affine: {
            int biasSize = TensorSize(op.B);
            int outer    = TensorSize(op.out) / (biasSize * op.inner);
            if (op.A.gradients)
                dispatch(affineBackwardDa, &op, outer, TensorSize(op.out));
            if (op.B.gradients)
                dispatch(affineBackwardDb, &op, biasSize, TensorSize(op.out));
            break;
        }
        case Linear: {
            Operation aff = { .type=Affine, .out=op.out, .A=op.product, .B=op.bias,
                              .u=op.u, .inner=op.inner };
            OperationBackward(aff);
            Operation mul = { .type=Multiply, .out=op.product, .A=op.A, .B=op.B };
            OperationBackward(mul);
            break;
        }
        case Convolution: {
            int Cout = op.B.shape[0];
            int K    = op.B.shape[1] * op.fH * op.fW;
            int S    = op.out.shape[2] * op.out.shape[3];
            int N    = op.out.shape[0];
            long gemmWork = (long)N * Cout * S * K;
            Operation aff = { .type=Affine, .out=op.out, .A=op.product, .B=op.bias,
                              .u=op.u, .inner=S };
            OperationBackward(aff);
            if (op.B.gradients) {
                int CK = Cout * K;
                op.scratch = TensorNew(2, (int[]){N, CK}, FALSE);
                dispatch(convolutionBackwardDb, &op, N, gemmWork);
                for (int n = 0; n < N; n++)
                    for (int j = 0; j < CK; j++)
                        op.B.gradients[j] += op.scratch.entries[(size_t)n * CK + j];
                TensorFree(op.scratch);
            }
            if (op.A.gradients)
                dispatch(convolutionBackwardDa, &op, N, gemmWork);
            break;
        }
        case Pool:
            if (op.A.gradients) {
                int planes = op.A.shape[0] * op.A.shape[1];
                dispatch(poolBackward, &op, planes, (long)TensorSize(op.A) * op.fH * op.fW);
            }
            break;
        case Unfold:
            if (op.A.gradients)
                dispatch(unfoldBackward, &op, op.A.shape[0] * op.A.shape[1], TensorSize(op.A));
            break;
        case Gather:
            if (op.A.gradients)
                dispatch(gatherBackward, &op, TensorSize(op.A), (long)TensorSize(op.A) * TensorSize(op.B));
            break;
        case Scatter:
            if (op.A.gradients)
                dispatch(scatterBackward, &op, TensorSize(op.out), (long)TensorSize(op.out) * TensorSize(op.B));
            break;
    }
}
