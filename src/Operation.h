#ifndef OPERATION_H
#define OPERATION_H

#include "tensorgrid.h"
#include "adt/ThreadPool.h"

// Represents an operation on a tensor, used by the autograd tape
typedef struct Operation Operation;
typedef void (*OperationFn)(Operation *op);

struct Operation {
    Tensor out, A, B;
    OperationFn forward, backward;
    void *context;
};

/*
 * Returns a new operation
 */
Operation OperationNew(Tensor out, Tensor A, Tensor B, OperationFn forward, OperationFn backward, void *context);

/*
 * Runs f over [0, total). Parallelizes 'work' with thread pool when it is above
 * a fixed threshold.
 */
void OperationDispatch(ThreadPoolFn f, Operation *op, int total, long work);

/*
 * Runs an operations forward pass and returns the output. If the output tensor
 * has autograd enabled, the operation is recorded on the autograd tape.
 */
Tensor OperationExecute(Operation op);

/*
 * The primitive operations every higher level tensor function is built from.
 */
Tensor OperationMap(Unary fn, Tensor A);
Tensor OperationZip(Binary fn, Tensor A, Tensor B);
Tensor OperationMultiply(Tensor A, Tensor B);
Tensor OperationConvolution(Tensor in, Tensor filter, int stride, int padding);
Tensor OperationPool(Binary fn, Entry initial, Tensor in, int size, int stride, int padding);
Tensor OperationGather(Tensor A, Tensor B, int axis);
Tensor OperationScatter(Binary fn, Tensor A, Tensor B, Entry fill, int axis, int n);

/*
 * Run the recorded autograd tape backwards from the loss tensor
 */
void OperationBackward(Tensor loss);


#endif
