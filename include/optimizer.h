#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "types.h"
#include "layer.h"
#include "backprop.h"

typedef struct Optimizer Optimizer;

typedef struct OptimizerState {

} OptimizerState;

typedef struct OptimizerAdditionalParams {

} OptimizerAdditionalParams;

typedef void (*OptimizerStep)(Optimizer *o, Layer *l, LayerGradients *grads);
typedef struct Optimizer {

    BASE_TYPE learning_rate;
    OptimizerState state;
    OptimizerAdditionalParams additional_params;
    OptimizerStep step;

} Optimizer;


Optimizer *create_SGD_optimizer(BASE_TYPE learning_rate);
void destroy_optimizer(Optimizer *o);

#endif