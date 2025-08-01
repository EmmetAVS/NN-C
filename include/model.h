#ifndef MODEL_H
#define MODEL_H

#include "types.h"
#include "layer.h"
#include "backprop.h"

typedef struct Optimizer Optimizer;

typedef struct Model {
    Layer **layers;
    size_t num_layers;
    LossFunction loss;

    //For backprop
    bool calc_grads;
    size_t max_grads;
    size_t current_grads_accumulated;
    LayerGradients ***gradients; //List of gradients that have been accumulated (LayerGradients*[num_layers][max_grads])

    //Stored averaged gradients (calculated by model_average_grads)
    LayerGradients **averaged_gradients;
} Model;

Model *create_model(size_t **shape, ActivationFunction *activations, size_t num_layers, LossFunction loss);
void destroy_model(Model *model);

void model_set_calculate_grads(Model *model, bool calc_grads);
void model_set_max_grads(Model *model, size_t max_grads);
void model_zero_grads(Model *model);

Vector *model_forward(Model *model, Vector *inputs);
void model_backward(Model *model, Vector *labels);

void model_average_grads(Model *model);
void model_clear_accumulated_grads(Model *model);

void model_step(Model *model, Optimizer *o);

#endif