#ifndef LAYER_H
#define LAYER_H

#include "types.h"
#include "activations.h"
#include "loss.h"

typedef struct Layer {

    Matrix *weights;
    Vector *biases;
    ActivationFunctionForward activation_forward;
    size_t input_size;
    size_t output_size;

} Layer;

Layer *create_layer(size_t input_size, size_t output_size, ActivationFunctionForward activation_forward);
void destroy_layer(Layer *l);
Vector *calculate_layer_outputs(Layer *l, Vector *inputs);

#endif