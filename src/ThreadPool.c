// https://github.com/alexjasson/tensorgrid

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <unistd.h>
#include "ThreadPool.h"

// Linux specific
#if defined(__linux__)

#include <limits.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>

#define FUTEX_WAIT    0
#define FUTEX_WAKE    1
#define FUTEX_PRIVATE 128
#define GUARD_SIZE    4096
#define STACK_SIZE    (2 * 1024 * 1024)

typedef struct { _Atomic int state; } Mutex;
typedef struct { _Atomic int seq;   } CondVar;

struct thread {
    _Atomic int  id;
    void        *base;
    void        *(*fn)(void *);
    void        *arg;
};

static int threadEntry(void *arg)
{
    struct thread *t = arg;
    t->fn(t->arg);
    syscall(SYS_exit, 0);
    return 0;
}

static struct thread *threadCreate(void *(*fn)(void *), void *arg)
{
    void *base = mmap(NULL, GUARD_SIZE + STACK_SIZE, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) return NULL;

    if (mprotect(base, GUARD_SIZE, PROT_NONE) == -1) {
        munmap(base, GUARD_SIZE + STACK_SIZE);
        return NULL;
    }

    struct thread *t = malloc(sizeof *t);
    if (!t) { munmap(base, GUARD_SIZE + STACK_SIZE); return NULL; }

    t->base = base;
    t->fn   = fn;
    t->arg  = arg;
    atomic_store_explicit(&t->id, 1, memory_order_relaxed);

    int flags = CLONE_VM | CLONE_FS | CLONE_FILES | CLONE_SIGHAND |
                CLONE_THREAD | CLONE_CHILD_CLEARTID;

    if (clone(threadEntry, (char *)base + GUARD_SIZE + STACK_SIZE, flags, t,
              NULL, NULL, (int *)&t->id) == -1) {
        free(t);
        munmap(base, GUARD_SIZE + STACK_SIZE);
        return NULL;
    }
    return t;
}

static void threadJoin(struct thread *t)
{
    int id;
    while ((id = atomic_load_explicit(&t->id, memory_order_acquire)) != 0)
        syscall(SYS_futex, (int *)&t->id, FUTEX_WAIT, id, NULL, NULL, 0);
    munmap(t->base, GUARD_SIZE + STACK_SIZE);
    free(t);
}

static void mutexLock(Mutex *m)
{
    int zero = 0;
    if (atomic_compare_exchange_strong_explicit(&m->state, &zero, 1,
            memory_order_acquire, memory_order_relaxed))
        return;
    while (atomic_exchange_explicit(&m->state, 2, memory_order_acquire) != 0)
        syscall(SYS_futex, (int *)&m->state, FUTEX_WAIT | FUTEX_PRIVATE, 2, NULL, NULL, 0);
}

static void mutexUnlock(Mutex *m)
{
    int old = atomic_exchange_explicit(&m->state, 0, memory_order_release);
    if (old == 2)
        syscall(SYS_futex, (int *)&m->state, FUTEX_WAKE | FUTEX_PRIVATE, 1, NULL, NULL, 0);
}

static void condVarWait(CondVar *c, Mutex *m)
{
    int seq = atomic_load_explicit(&c->seq, memory_order_relaxed);
    mutexUnlock(m);
    syscall(SYS_futex, (int *)&c->seq, FUTEX_WAIT | FUTEX_PRIVATE, seq, NULL, NULL, 0);
    mutexLock(m);
}

static void condVarSignal(CondVar *c)
{
    atomic_fetch_add_explicit(&c->seq, 1, memory_order_relaxed);
    syscall(SYS_futex, (int *)&c->seq, FUTEX_WAKE | FUTEX_PRIVATE, 1, NULL, NULL, 0);
}

static void condVarBroadcast(CondVar *c)
{
    atomic_fetch_add_explicit(&c->seq, 1, memory_order_relaxed);
    syscall(SYS_futex, (int *)&c->seq, FUTEX_WAKE | FUTEX_PRIVATE, INT_MAX, NULL, NULL, 0);
}

// Mac specific
#elif defined(__APPLE__)

#include <pthread.h>

typedef struct { pthread_mutex_t m; } Mutex;
typedef struct { pthread_cond_t c;  } CondVar;

struct thread {
    pthread_t tid;
};

static struct thread *threadCreate(void *(*fn)(void *), void *arg)
{
    struct thread *t = malloc(sizeof *t);
    if (!t) return NULL;
    if (pthread_create(&t->tid, NULL, fn, arg) != 0) {
        free(t);
        return NULL;
    }
    return t;
}

static void threadJoin(struct thread *t)
{
    pthread_join(t->tid, NULL);
    free(t);
}

static void mutexLock(Mutex *m)               { pthread_mutex_lock(&m->m);       }
static void mutexUnlock(Mutex *m)             { pthread_mutex_unlock(&m->m);     }
static void condVarWait(CondVar *c, Mutex *m) { pthread_cond_wait(&c->c, &m->m); }
static void condVarSignal(CondVar *c)         { pthread_cond_signal(&c->c);      }
static void condVarBroadcast(CondVar *c)      { pthread_cond_broadcast(&c->c);   }

#else
#error "Unsupported OS! Only linux and mac are supported"
#endif

// Both

static ThreadPool newThreadPool(int n);
static void freeThreadPool(void);
static void *threadFn(void *arg);

struct threadPool {
    struct thread **threads;
    int numThreads;
    Mutex mutex;
    CondVar mapReady;
    CondVar mapDone;
    KernelFn f;
    void *mapArgs;
    int mapTotal;
    int mapId;
    int subsetsDone;
    int shutdown;
};

static ThreadPool newThreadPool(int n)
{
    ThreadPool pool = malloc(sizeof(*pool));
    if (!pool) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    pool->numThreads  = n;
    pool->mapId       = 0;
    pool->shutdown    = 0;

#if defined(__linux__)
    pool->mutex    = (Mutex){0};
    pool->mapReady = (CondVar){0};
    pool->mapDone  = (CondVar){0};
#elif defined(__APPLE__)
    pthread_mutex_init(&pool->mutex.m, NULL);
    pthread_cond_init(&pool->mapReady.c, NULL);
    pthread_cond_init(&pool->mapDone.c, NULL);
#endif

    pool->threads = malloc(n * sizeof(struct thread *));
    if (!pool->threads) {
        fprintf(stderr, "Insufficient memory!\n");
        exit(EXIT_FAILURE);
    }

    return pool;
}

static void freeThreadPool(void)
{
    ThreadPool pool = ThreadPoolGet();

    mutexLock(&pool->mutex);
    pool->shutdown = 1;
    condVarBroadcast(&pool->mapReady);
    mutexUnlock(&pool->mutex);

    for (int i = 0; i < pool->numThreads; i++)
        threadJoin(pool->threads[i]);

#if defined(__APPLE__)
    pthread_mutex_destroy(&pool->mutex.m);
    pthread_cond_destroy(&pool->mapReady.c);
    pthread_cond_destroy(&pool->mapDone.c);
#endif

    free(pool->threads);
    free(pool);
}

ThreadPool ThreadPoolGet(void)
{
    static ThreadPool pool;
    if (!pool) {
        pool = newThreadPool((int)sysconf(_SC_NPROCESSORS_ONLN));
        for (int i = 0; i < pool->numThreads; i++) {
            pool->threads[i] = threadCreate(threadFn, (void *)(uintptr_t)i);
            if (!pool->threads[i]) {
                fprintf(stderr, "Failed to create worker thread!\n");
                exit(EXIT_FAILURE);
            }
        }
        atexit(freeThreadPool);
    }
    return pool;
}

void ThreadPoolMap(ThreadPool pool, KernelFn f, void *arg, int total)
{
    mutexLock(&pool->mutex);
    pool->f           = f;
    pool->mapArgs     = arg;
    pool->mapTotal    = total;
    pool->subsetsDone = 0;
    pool->mapId++;
    condVarBroadcast(&pool->mapReady);

    while (pool->subsetsDone < pool->numThreads)
        condVarWait(&pool->mapDone, &pool->mutex);
    mutexUnlock(&pool->mutex);
}

static void *threadFn(void *arg)
{
    int id = (int)(uintptr_t)arg;
    ThreadPool pool = ThreadPoolGet();
    int lastMapId = 0;

    while (1) {
        mutexLock(&pool->mutex);
        while (pool->mapId == lastMapId && !pool->shutdown)
            condVarWait(&pool->mapReady, &pool->mutex);

        if (pool->shutdown) {
            mutexUnlock(&pool->mutex);
            return NULL;
        }

        lastMapId = pool->mapId;
        KernelFn f = pool->f;
        void *ma   = pool->mapArgs;
        int total  = pool->mapTotal;
        int n      = pool->numThreads;
        mutexUnlock(&pool->mutex);

        int subsetSize = total / n;
        int remainder  = total % n;
        int start = id * subsetSize + (id < remainder ? id : remainder);
        int end   = start + subsetSize + (id < remainder ? 1 : 0);

        if (start < end)
            f(ma, start, end);

        mutexLock(&pool->mutex);
        pool->subsetsDone++;
        if (pool->subsetsDone == n)
            condVarSignal(&pool->mapDone);
        mutexUnlock(&pool->mutex);
    }
}
