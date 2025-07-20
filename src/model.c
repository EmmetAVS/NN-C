#include "model.h"
#include <stdlib.h>

Model *create_model(size_t **shape, ActivationFunction *activations, size_t num_layers, LossFunction loss) {

    Model *model = (Model *)malloc(sizeof(Model));
    model->num_layers = num_layers;
    model->layers = (Layer **)malloc(sizeof(Layer *) * num_layers);
    for (size_t i = 0; i < num_layers; i ++) {

        model->layers[i] = create_layer(shape[i][0], shape[i][1], activations[i]);

    }

    model->loss = loss;
    model->calc_grads = true;
    model->current_grads_calculated = 0;
    model->gradients = NULL;

    return model;

}

void destroy_model(Model *model) {

    model_zero_grads(model);

    for (size_t i = 0; i < model->num_layers; i ++) {

        destroy_layer(model->layers[i]);

    }

    free(model->layers);
    free(model);

}

void model_set_calculate_grads(Model *model, bool calc_grads) {

    model->calc_grads = calc_grads;

}

void model_zero_grads(Model *model) {

    if (!model->gradients) return;
    for (size_t i = 0; i < model->current_grads_calculated; i ++) {

        destroy_layer_gradients(model->gradients[i]);

    }

    free(model->gradients);
    model->current_grads_calculated = 0;

}

static Vector *model_inference(Model *model, Vector *inputs) {

    //Assume model no grad for inference

    Vector *current_output = inputs;

    for (size_t i = 0; i < model->num_layers; i++) {

        current_output = forward_layer(model->layers[i], current_output, false);

    }

    return current_output;

}

static Vector *model_forward_with_grad(Model *model, Vector *inputs) {

    Vector *current_output = inputs;

    for (size_t i = 0; i < model->num_layers; i++) {

        current_output = forward_layer(model->layers[i], current_output, true);

    }

    return current_output;

}

Vector *model_forward(Model *model, Vector *inputs) {

    if (model->calc_grads) {
        return model_inference(model, inputs);
    }

    return model_forward_with_grad(model, inputs);

}

void model_backward(Model *model) {

    if (!model->calc_grads) return;
    
    

}