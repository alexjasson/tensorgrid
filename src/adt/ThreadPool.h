#ifndef THREADPOOL_H
#define THREADPOOL_H

typedef void (*ThreadPoolFn)(void *arg, int start, int end);
typedef struct threadPool *ThreadPool;

/*
 * Singleton pattern to initialize a thread pool, created on first use and destroyed at exit.
 */
ThreadPool ThreadPoolGet(void);

/*
 * Divides total items into subsets, maps f across total threads and blocks until complete
 */
void ThreadPoolMap(ThreadPool pool, ThreadPoolFn f, void *arg, int total);

/*
 * The number of worker threads the pool maps across.
 */
int ThreadPoolSize(ThreadPool pool);

/*
 * Acquire/release the pool's lock for short critical sections. Used when threads need to
 * synchronize on shared data. 
 */
void ThreadPoolLock(ThreadPool pool);
void ThreadPoolUnlock(ThreadPool pool);

#endif
