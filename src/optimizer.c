#include "optimizer.h"
#include <stdlib.h>

static void optimizer_step_sgd(Optimizer *o, Layer *l, LayerGradients *grads) {

    update_layer_parameters(l, grads, o->learning_rate);

}

Optimizer *create_SGD_optimizer(BASE_TYPE learning_rate) {
    
    Optimizer* o = (Optimizer *)malloc(sizeof(Optimizer));
    o->learning_rate = learning_rate;
    o->step = optimizer_step_sgd;

    return o;
}

void destroy_optimizer(Optimizer *o) {

    free(o);

}