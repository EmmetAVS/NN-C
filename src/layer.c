#include "layer.h"
#include "operations.h"
#include <stdlib.h>

Layer *create_layer(size_t input_size, size_t output_size, ActivationFunction activation) {

    Layer *l = (Layer *)malloc(sizeof(Layer));

    l->activation = activation;
    l->input_size = input_size;
    l->output_size = output_size;

    l->biases = create_vector(output_size);
    l->weights = create_matrix(output_size, input_size);

    BASE_TYPE weight_scale = sqrt(2.0 / (input_size + output_size));
    
    for (size_t i = 0; i < output_size; i++) {
        for (size_t j = 0; j < input_size; j++) {
            BASE_TYPE random_val = ((BASE_TYPE)rand() / RAND_MAX) * 2.0 - 1.0;
            matrix_set_value_at(l->weights, i, j, random_val * weight_scale);
        }
    }

    l->context = NULL;
    return l;

}

void destroy_layer(Layer *l) {

    if (!l)
        return;

    destroy_matrix(l->weights);
    destroy_vector(l->biases);

    if (l->context) destroy_layer_context(l->context);

    free(l);

}

LayerContext *create_layer_context(Vector *inputs, Vector *logits, Vector *activated) {

    LayerContext *context = (LayerContext *)malloc(sizeof(LayerContext));

    context->inputs = inputs;
    context->logits = logits;
    context->activated_output = activated;

    return context;

}

void destroy_layer_context(LayerContext *context) {

    /*
    Only logits needs to be destroyed, as the inputs and activated vectors are both accessible elsewhere, and therefore 
    must be handled elsewhere
    */
    destroy_vector(context->logits);
    
    free(context);

}