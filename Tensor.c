// https://github.com/alexjasson/tensor

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "Tensor.h"

// Kernel function that operates on a subset of entries [start, end)
typedef void (*KernelFn)(void *arg, int start, int end);
typedef struct {
    Tensor out, A;
    UnaryFn f;
} UnaryArgs;
typedef struct {
    Tensor out, A, B;
    BinaryFn f;
} BinaryArgs;

// Thread pool representation
typedef struct threadPool {
    pthread_t *threads;
    int numThreads;
    pthread_mutex_t mutex;
    pthread_cond_t mapReady;
    pthread_cond_t mapDone;
    KernelFn f;
    void *mapArgs;
    int mapTotal;
    int mapId;
    int subsetsDone;
    int shutdown;
} *ThreadPool;

// Kernel functions
static void mapSubset(void *arg, int start, int end);
static void zipSubset(void *arg, int start, int end);
static void multiplySubset(void *arg, int start, int end);

// Thread pool functions
static ThreadPool threadPoolNew(int n);
static void threadPoolFree(void);
static ThreadPool threadPoolGet(void);
static void threadPoolMap(ThreadPool pool, KernelFn f, void *arg, int total);
static void *threadFn(void *arg);

// Utility functions
static float randomFloat(float lower, float upper);

/*
 * Tensor implementation
 */
int TensorSize(Tensor A)
{
    int size = 1;
    for (int i = 0; i < A.dimensions; i++)
        size *= A.shape[i];
    return size;
}

Tensor TensorNew(int dimensions, int *shape)
{
    Tensor t;
    t.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++)
        t.shape[i] = shape[i];
    t.strides[dimensions - 1] = 1;
    for (int i = dimensions - 2; i >= 0; i--)
        t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
    t.entries = calloc(TensorSize(t), sizeof(Entry));
    if (!t.entries) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }
    return t;
}

void TensorFree(Tensor A)
{
    free(A.entries);
}

Tensor TensorView(int dimensions, int *shape, Entry *entries)
{
    Tensor t;
    t.dimensions = dimensions;
    for (int i = 0; i < dimensions; i++)
        t.shape[i] = shape[i];
    t.strides[dimensions - 1] = 1;
    for (int i = dimensions - 2; i >= 0; i--)
        t.strides[i] = t.strides[i + 1] * t.shape[i + 1];
    t.entries = entries;
    return t;
}

Tensor TensorSlice(Tensor A, int index)
{
    Tensor t;
    t.dimensions = A.dimensions - 1;
    for (int i = 0; i < t.dimensions; i++) {
        t.shape[i]   = A.shape[i + 1];
        t.strides[i] = A.strides[i + 1];
    }
    t.entries = A.entries + index * A.strides[0];
    return t;
}

Tensor TensorReshape(Tensor A, int dimensions, int *shape)
{
    return TensorView(dimensions, shape, A.entries);
}

Tensor TensorTranspose(Tensor A)
{
    Tensor t = A;
    t.shape[A.dimensions - 2]   = A.shape[A.dimensions - 1];
    t.shape[A.dimensions - 1]   = A.shape[A.dimensions - 2];
    t.strides[A.dimensions - 2] = A.strides[A.dimensions - 1];
    t.strides[A.dimensions - 1] = A.strides[A.dimensions - 2];
    return t;
}

Entry TensorReduce(BinaryFn f, Entry initial, Tensor A)
{
    Entry result = initial;
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        result = f(result, A.entries[i]);
    return result;
}

void TensorMap(Tensor out, UnaryFn f, Tensor A)
{
    UnaryArgs args = {out, A, f};
    threadPoolMap(threadPoolGet(), mapSubset, &args, TensorSize(A));
}

void TensorZip(Tensor out, BinaryFn f, Tensor A, Tensor B)
{
    BinaryArgs args = {out, A, B, f};
    threadPoolMap(threadPoolGet(), zipSubset, &args, TensorSize(A));
}

void TensorMultiply(Tensor out, Tensor A, Tensor B)
{
    int m = A.shape[A.dimensions - 2];
    int k = A.shape[A.dimensions - 1];
    int batchSize = TensorSize(A) / (m * k);
    BinaryArgs args = {out, A, B, NULL};
    threadPoolMap(threadPoolGet(), multiplySubset, &args, batchSize * m);
}

void TensorSoftmax(Tensor A)
{
    int n     = A.shape[A.dimensions - 1];
    int batch = TensorSize(A) / n;
    for (int i = 0; i < batch; i++) {
        Tensor row = TensorSlice(A, i);
        Entry max  = TensorReduce(fmaxf, row.entries[0], row);
        for (int j = 0; j < n; j++)
            row.entries[j] = expf(row.entries[j] - max);
        Entry sum  = TensorReduce(add, 0.0f, row);
        for (int j = 0; j < n; j++)
            row.entries[j] /= sum;
    }
}

void TensorXavier(Tensor A)
{
    Entry bound = sqrtf(6.0f / (A.shape[A.dimensions - 2] + A.shape[A.dimensions - 1]));
    int size = TensorSize(A);
    for (int i = 0; i < size; i++)
        A.entries[i] = (Entry)randomFloat(-bound, bound);
}


Entry TensorGet(Tensor A, int *indices)
{
    int idx = 0;
    for (int i = 0; i < A.dimensions; i++)
        idx += indices[i] * A.strides[i];
    return A.entries[idx];
}

void TensorSet(Tensor A, int *indices, Entry value)
{
    int idx = 0;
    for (int i = 0; i < A.dimensions; i++)
        idx += indices[i] * A.strides[i];
    A.entries[idx] = value;
}

static void mapSubset(void *arg, int start, int end)
{
    UnaryArgs *args = arg;
    for (int i = start; i < end; i++)
        args->out.entries[i] = args->f(args->A.entries[i]);
}

static void zipSubset(void *arg, int start, int end)
{
    BinaryArgs *args = arg;
    for (int i = start; i < end; i++)
        args->out.entries[i] = args->f(args->A.entries[i], args->B.entries[i]);
}

static void multiplySubset(void *arg, int start, int end)
{
    BinaryArgs *args = arg;
    int m = args->A.shape[args->A.dimensions - 2];
    int k = args->A.shape[args->A.dimensions - 1];
    int n = args->B.shape[1];
    int strideA_row = args->A.strides[args->A.dimensions - 2];
    int strideA_col = args->A.strides[args->A.dimensions - 1];
    int strideB_row = args->B.strides[0];
    int strideB_col = args->B.strides[1];
    int batchStride = (args->A.dimensions >= 3) ? args->A.strides[args->A.dimensions - 3] : 0;

    for (int i = start; i < end; i++) {
        int batch = i / m;
        int row   = i % m;

        Entry *A_batch = args->A.entries + batch * batchStride;
        Entry *C_row   = args->out.entries + (batch * m + row) * n;

        for (int j = 0; j < n; j++)
            C_row[j] = 0.0f;

        for (int l = 0; l < k; l++) {
            Entry a = A_batch[row * strideA_row + l * strideA_col];
            for (int j = 0; j < n; j++)
                C_row[j] += a * args->B.entries[l * strideB_row + j * strideB_col];
        }
    }
}

/*
 * Thread pool implementation
 */
static void *threadFn(void *arg)
{
    int id = (int)(uintptr_t)arg;
    ThreadPool pool = threadPoolGet();
    int lastMapId = 0;

    while (1) {
        pthread_mutex_lock(&pool->mutex);
        while (pool->mapId == lastMapId && !pool->shutdown)
            pthread_cond_wait(&pool->mapReady, &pool->mutex);

        if (pool->shutdown) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }

        lastMapId = pool->mapId;
        KernelFn f = pool->f;
        void *ma = pool->mapArgs;
        int total = pool->mapTotal;
        int n = pool->numThreads;
        pthread_mutex_unlock(&pool->mutex);

        int subsetSize = total / n;
        int remainder = total % n;
        int start = id * subsetSize + (id < remainder ? id : remainder);
        int end = start + subsetSize + (id < remainder ? 1 : 0);

        if (start < end)
            f(ma, start, end);

        pthread_mutex_lock(&pool->mutex);
        pool->subsetsDone++;
        if (pool->subsetsDone == n)
            pthread_cond_signal(&pool->mapDone);
        pthread_mutex_unlock(&pool->mutex);
    }
}

static ThreadPool threadPoolNew(int n)
{
    ThreadPool pool = malloc(sizeof(struct threadPool));
    if (!pool) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    pool->numThreads = n;
    pool->mapId = 0;
    pool->shutdown = 0;

    pthread_mutex_init(&pool->mutex, NULL);
    pthread_cond_init(&pool->mapReady, NULL);
    pthread_cond_init(&pool->mapDone, NULL);

    pool->threads = malloc(n * sizeof(pthread_t));
    if (!pool->threads) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    return pool;
}

static void threadPoolFree(void)
{
    ThreadPool pool = threadPoolGet();

    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->mapReady);
    pthread_mutex_unlock(&pool->mutex);

    for (int i = 0; i < pool->numThreads; i++)
        pthread_join(pool->threads[i], NULL);

    pthread_mutex_destroy(&pool->mutex);
    pthread_cond_destroy(&pool->mapReady);
    pthread_cond_destroy(&pool->mapDone);
    free(pool->threads);
    free(pool);
}

static ThreadPool threadPoolGet(void)
{
    static ThreadPool pool;
    if (!pool) {
        pool = threadPoolNew((int)sysconf(_SC_NPROCESSORS_ONLN));
        for (int i = 0; i < pool->numThreads; i++)
            pthread_create(&pool->threads[i], NULL, threadFn, (void *)(uintptr_t)i);
        atexit(threadPoolFree);
    }
    return pool;
}

static void threadPoolMap(ThreadPool pool, KernelFn f, void *arg, int total)
{
    pthread_mutex_lock(&pool->mutex);
    pool->f = f;
    pool->mapArgs = arg;
    pool->mapTotal = total;
    pool->subsetsDone = 0;
    pool->mapId++;
    pthread_cond_broadcast(&pool->mapReady);

    while (pool->subsetsDone < pool->numThreads)
        pthread_cond_wait(&pool->mapDone, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

/*
 * Utility functions
 */
static uint32_t xorshift(void)
{
    static uint32_t state = 0;
    static int seeded = 0;

    if (!seeded) {
        state = (uint32_t)time(NULL);
        seeded = 1;
    }

    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

static float randomFloat(float lower, float upper)
{
    float r = (xorshift() >> 8) / 16777216.0f;
    return lower + r * (upper - lower);
}
