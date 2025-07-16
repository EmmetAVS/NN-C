#ifndef BACKPROP_H
#define BACKPROP_H

#include "types.h"
#include "activations.h"
#include "layer.h"
#include "loss.h"

typedef struct LayerGradients {
    Matrix *d_weights;
    Vector *d_biases;
    Vector *d_inputs;
} LayerGradients;

typedef enum BackpropContextType {

    dLoss_dActivation, LabelsOutput

} BackpropContextType;

typedef struct BackpropContextDLossDActivation {
    Vector *dL_dA;
    Vector *activated;
} BackpropContextDLossDActivation;

typedef struct BackpropContextLabelsOutput {
    Vector *labels;
} BackpropContextLabelsOutput;

typedef struct BackpropContext {

    BackpropContextType type;
    BackpropContextDLossDActivation dLoss_dActivation;
    BackpropContextLabelsOutput labels_output;

} BackpropContext;

Vector *forward_layer(Layer *layer, Vector *input);

LayerGradients *backward_layer(Layer *layer, Vector *input, Vector *logits, BackpropContext *context);

void update_layer_parameters(Layer *layer, LayerGradients *grads, BASE_TYPE learning_rate);

LayerGradients *create_layer_gradients(size_t input_size, size_t output_size);
void destroy_layer_gradients(LayerGradients *grads);

#endif
