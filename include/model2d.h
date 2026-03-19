#ifndef MODEL2D_H
#define MODEL2D_H

#include "types.h"
#include "layer.h"
#include "backprop.h"

typedef struct Optimizer2D Optimizer2D;

typedef struct Model2D {
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
} Model2D;

Model2D *create_model2d(size_t **shape, ActivationFunction *activations, size_t num_layers, LossFunction loss);
void destroy_model2d(Model2D *model);

void model2d_set_calculate_grads(Model2D *model, bool calc_grads);
void model2d_set_max_grads(Model2D *model, size_t max_grads);
void model2d_zero_grads(Model2D *model);

Vector *model2d_forward(Model2D *model, Vector *inputs);
void model2d_backward(Model2D *model, Vector *labels);

void model2d_average_grads(Model2D *model);
void model2d_clear_accumulated_grads(Model2D *model);

void model2d_step(Model2D *model, Optimizer2D *o);

#endif