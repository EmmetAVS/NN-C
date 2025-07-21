#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "types.h"

typedef struct OptimizerState {

} OptimizerState;

typedef struct OptimizerAdditionalParams {

} OptimizerAdditionalParams;

typedef struct Optimizer {

    BASE_TYPE learning_rate;
    OptimizerState state;
    OptimizerAdditionalParams additional_params;

} Optimizer;

Optimizer *create_SGD_optimizer(BASE_TYPE learning_rate);
void destroy_optimizer(Optimizer *o);

#endif