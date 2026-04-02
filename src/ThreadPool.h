#ifndef THREADPOOL_H
#define THREADPOOL_H

typedef void (*KernelFn)(void *arg, int start, int end);
typedef struct threadPool *ThreadPool;

ThreadPool ThreadPoolGet(void);
void ThreadPoolMap(ThreadPool pool, KernelFn f, void *arg, int total);

#endif
