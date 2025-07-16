#ifndef LAYER_H
#define LAYER_H

#include "types.h"
#include "activations.h"
#include "loss.h"

typedef struct Layer {

    Matrix *weights;
    Vector *biases;
    ActivationFunction activation;
    size_t input_size;
    size_t output_size;

} Layer;

Layer *create_layer(size_t input_size, size_t output_size, ActivationFunction activation);
void destroy_layer(Layer *l);

#endif