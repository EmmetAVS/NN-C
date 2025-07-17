#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "backprop.h"
#include "tests.h"

void test_forward_backward_update() {
    Vector *input = create_vector(3);
    input->data[0] = 1.0;
    input->data[1] = 2.0;
    input->data[2] = 3.0;

    Vector *label = create_vector(2);
    label->data[0] = 0.0;
    label->data[1] = 1.0;

    Layer *layer = create_layer(3, 2, activation_loss_softmax_cross_entropy);

    for (size_t i = 0; i < 3 * 2; i++) {
        layer->weights->data[i] = 0.1f;
    }
    for (size_t i = 0; i < 2; i++) {
        layer->biases->data[i] = 0.0f;
    }

    Vector *logits = forward_layer(layer, input);
    assert(logits != NULL);
    assert(logits->length == 2);

    BackpropContext context = {
        .type = LabelsOutput,
        .labels_output.labels = label
    };

    LayerGradients *grads = backward_layer(layer, input, logits, &context);
    assert(grads != NULL);
    assert(grads->d_weights != NULL);
    assert(grads->d_biases != NULL);
    assert(grads->d_inputs != NULL);

    BASE_TYPE old_weight = layer->weights->data[0];
    BASE_TYPE learning_rate = 0.01;

    update_layer_parameters(layer, grads, learning_rate);
    BASE_TYPE new_weight = layer->weights->data[0];

    assert(fabs(new_weight - old_weight) > 0.0f);

    //Cleanup
    destroy_vector(input);
    destroy_vector(label);
    destroy_vector(logits);
    destroy_layer_gradients(grads);
    destroy_layer(layer);
}

void test_create_destroy_layer_gradients() {
    LayerGradients *grads = create_layer_gradients(3, 2);
    assert(grads != NULL);
    assert(grads->d_weights != NULL);
    assert(grads->d_biases != NULL);
    assert(grads->d_inputs != NULL);
    destroy_layer_gradients(grads);
}

int main() {
    test_create_destroy_layer_gradients();
    test_forward_backward_update();
    return 0;
}
