#ifndef LAYER_H
#define LAYER_H

#include "types.h"
#include "activations.h"
#include "loss.h"
#include "backprop.h"

typedef struct LayerContext {

    Vector *inputs;
    Vector *logits;
    Vector *activated_output;

} LayerContext;

typedef struct Layer {

    Matrix *weights;
    Vector *biases;
    ActivationFunction activation;
    size_t input_size;
    size_t output_size;
    LayerContext *context;
    LayerGradients *grads;

} Layer;

Layer *create_layer(size_t input_size, size_t output_size, ActivationFunction activation);
void destroy_layer(Layer *l);
LayerContext *create_layer_context(Vector *inputs, Vector *logits, Vector *activated);
void destroy_layer_context(LayerContext *context);

#endif