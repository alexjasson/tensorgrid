#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "tensorgrid.h"

/*
 * A registry of a model's trainable Tensors
 */
typedef struct parameters *Parameters;

/*
 * Creates an empty parameter registry
 */
Parameters ParametersNew(void);

/*
 * Allocates a new autograd-enabled tensor, registers it in the collection and
 * returns it.
 */
Tensor ParametersAdd(Parameters p, int dimensions, int *shape);

/*
 * Adds the total number of entries in each Tensor and returns it
 */
int ParametersCount(Parameters p);

/*
 * Applies a single optimizer step to every registered tensor
 */
void ParametersSGD(Parameters p, float lr, float momentum);
void ParametersAdam(Parameters p, float lr, float beta1, float beta2, float epsilon, float weightDecay);

/*
 * Frees every registered tensor and the registry itself
 */
void ParametersFree(Parameters p);

#endif
