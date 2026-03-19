#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "types.h"
#include "layer.h"
#include "backprop.h"
#include "model2d.h"

typedef struct Optimizer2D Optimizer2D;

typedef void (*Optimizer2DStep)(Optimizer2D *o, Layer *l, LayerGradients *grads);
typedef void (*Optimizer2DDestructionFunc)(Optimizer2D *o);
typedef struct Optimizer2D {

    BASE_TYPE learning_rate;
    Optimizer2DStep step;
    Optimizer2DDestructionFunc destruction;

} Optimizer2D;

typedef struct AdamOptimizer2D {
    Optimizer2D opt;

    //Params
    BASE_TYPE beta_1;
    BASE_TYPE beta_2;

    //State
    Matrix **weight_m_t;
    Matrix **weight_v_t;
    Vector **bias_m_t;
    Vector **bias_v_t;
    size_t layers;
    Model2D *model;
    size_t *steps;

} AdamOptimizer2D;

Optimizer2D *create_SGD_optimizer2d(BASE_TYPE learning_rate);
Optimizer2D *create_ADAM_optimizer2d(Model2D *model, BASE_TYPE learning_rate, BASE_TYPE beta_1, BASE_TYPE beta_2);
void destroy_optimizer2d(Optimizer2D *o);

#endif