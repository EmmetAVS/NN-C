#include "optimizer.h"
#include <stdlib.h>

Optimizer *create_SGD_optimizer(BASE_TYPE learning_rate) {
    
    Optimizer* o = (Optimizer *)malloc(sizeof(Optimizer));
    o->learning_rate = learning_rate;

    return o;
}

void destroy_optimizer(Optimizer *o) {

    free(o);

}