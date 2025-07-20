#ifndef MODEL_H
#define MODEL_H

#include "types.h"
#include "layer.h"

typedef struct Model {
    Layer **layers;
    size_t num_layers;
    LossFunction loss;
} Model;

#endif