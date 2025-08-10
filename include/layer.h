#ifndef LAYER_H
#define LAYER_H

#include "types.h"
#include "activations.h"
#include "loss.h"

typedef struct BackpropContext BackpropContext;
typedef struct LayerGradients LayerGradients;
typedef struct Layer Layer;

typedef Vector * (*LayerForward)(Layer *layer, Vector *input, bool save_context);
typedef LayerGradients * (*LayerBackward)(Layer *layer, Vector *input, Vector *logits, BackpropContext *context); 

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
    LayerForward forward;
    LayerBackward backward;

} Layer;

Layer *create_layer(size_t input_size, size_t output_size, ActivationFunction activation);
void destroy_layer(Layer *l);
LayerContext *create_layer_context(Vector *inputs, Vector *logits, Vector *activated);
void destroy_layer_context(LayerContext *context);

#endif