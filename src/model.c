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

    return model;

}

void destroy_model(Model *model) {

    for (size_t i = 0; i < model->num_layers; i ++) {

        destroy_layer(model->layers[i]);

    }

    free(model->layers);
    free(model);

}

Vector *