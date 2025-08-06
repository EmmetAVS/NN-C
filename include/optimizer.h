#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "types.h"
#include "layer.h"
#include "backprop.h"
#include "model.h"

typedef struct Optimizer Optimizer;

typedef void (*OptimizerStep)(Optimizer *o, Layer *l, LayerGradients *grads);
typedef void (*OptimizerDestructionFunc)(Optimizer *o);
typedef struct Optimizer {

    BASE_TYPE learning_rate;
    OptimizerStep step;
    void *additional_parameters;
    void *optimizer_state;
    OptimizerDestructionFunc destruction;

} Optimizer;

typedef struct OptimizerAdamParams {

    BASE_TYPE beta_1;
    BASE_TYPE beta_2;

} OptimizerAdamParams;

typedef struct OptimizerAdamState {

    Matrix **weight_m_t;
    Matrix **weight_v_t;
    Vector **bias_m_t;
    Vector **bias_v_t;
    size_t layers;
    Model *model;
    size_t *steps;

} OptimizerAdamState;

Optimizer *create_SGD_optimizer(BASE_TYPE learning_rate);
Optimizer *create_ADAM_optimizer(Model *model, BASE_TYPE learning_rate, BASE_TYPE beta_1, BASE_TYPE beta_2);
void destroy_optimizer(Optimizer *o);

#endif