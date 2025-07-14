#include "layer.h"
#include "operations.h"
#include <stdlib.h>

Layer *create_layer(size_t input_size, size_t output_size, ActivationFunctionForward activation_forward) {

    Layer *l = (Layer *)malloc(sizeof(Layer));

    l->activation_forward = activation_forward;
    l->input_size = input_size;
    l->output_size = output_size;

    l->biases = create_vector(output_size);
    l->weights = create_matrix(output_size, input_size);

    return l;

}

void destroy_layer(Layer *l) {

    if (!l)
        return;

    destroy_matrix(l->weights);
    destroy_vector(l->biases);

    free(l);

}

Vector *calculate_layer_outputs(Layer *l, Vector *inputs) {

    Vector *logits_no_bias = multiply_matrix_with_vector(l->weights, inputs);
    Vector *logits = add_vector_to_vector(logits_no_bias, l->biases);
    destroy_vector(logits_no_bias);

    Vector *activated = (l->activation_forward)(logits);
    destroy_vector(logits);

    return activated;

}