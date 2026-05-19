#include <stdlib.h>
#include "Parameters.h"
#include "DynamicArray.h"

struct parameters {
    DynamicArray tensors;
};

Parameters ParametersNew(void)
{
    Parameters p = malloc(sizeof(struct parameters));
    p->tensors = DynamicArrayNew(sizeof(Tensor));
    return p;
}

Tensor ParametersAdd(Parameters p, int dimensions, int *shape)
{
    Tensor t = TensorNew(dimensions, shape, TRUE);
    DynamicArrayPush(p->tensors, &t);
    return t;
}

int ParametersCount(Parameters p)
{
    int total = 0;
    for (int i = 0; i < DynamicArraySize(p->tensors); i++)
        total += TensorSize(*(Tensor *)DynamicArrayGet(p->tensors, i));
    return total;
}

void ParametersSGD(Parameters p, float lr, float momentum)
{
    for (int i = 0; i < DynamicArraySize(p->tensors); i++)
        TensorSGD(*(Tensor *)DynamicArrayGet(p->tensors, i), lr, momentum);
}

void ParametersAdam(Parameters p, float lr, float beta1, float beta2, float epsilon, float weightDecay)
{
    for (int i = 0; i < DynamicArraySize(p->tensors); i++)
        TensorAdam(*(Tensor *)DynamicArrayGet(p->tensors, i), lr, beta1, beta2, epsilon, weightDecay);
}

void ParametersFree(Parameters p)
{
    for (int i = 0; i < DynamicArraySize(p->tensors); i++)
        TensorFree(*(Tensor *)DynamicArrayGet(p->tensors, i));
    DynamicArrayFree(p->tensors);
    free(p);
}
