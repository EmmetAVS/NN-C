#ifndef MODEL_H
#define MODEL_H

#include "types.h"
#include "layer.h"
#include "backprop.h"

typedef struct Model {
    Layer **layers;
    size_t num_layers;
    LossFunction loss;

    //For backprop
    bool calc_grads;
    size_t current_grads_calculated;
    LayerGradients **gradients;
} Model;
#include "model.h"
#include <stdlib.h>

Model *create_model(size_t **shape, ActivationFunction *activations, size_t num_layers, LossFunction loss);

void destroy_model(Model *model);

void model_set_calculate_grads(Model *model, bool calc_grads);

void model_zero_grads(Model *model);

Vector *model_forward(Model *model, Vector *inputs);

#endif