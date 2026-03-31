// https://github.com/alexjasson/tensorgrid

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <pthread.h>
#include "ThreadPool.h"

struct threadPool {
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
};

static ThreadPool threadPoolNew(int n);
static void threadPoolFree(void);
static void *threadFn(void *arg);

static ThreadPool threadPoolNew(int n)
{
    ThreadPool pool = malloc(sizeof(*pool));
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

ThreadPool threadPoolGet(void)
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

void threadPoolMap(ThreadPool pool, KernelFn f, void *arg, int total)
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
