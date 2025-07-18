#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "backprop.h"
#include "tests.h"

void test_forward_backward_update() {
    Vector *input = create_vector(3);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;

    Vector *label = create_vector(2);
    label->data[0] = 0.0f;
    label->data[1] = 1.0f;

    Layer *layer = create_layer(3, 2, activation_loss_softmax_cross_entropy);

    // Initialize weights and biases
    for (size_t i = 0; i < 3 * 2; i++) {
        layer->weights->data[i] = 0.1f;
    }
    for (size_t i = 0; i < 2; i++) {
        layer->biases->data[i] = 0.0f;
    }

    printf("Layer created, weights set, biases set\n");

    Vector *logits = forward_layer(layer, input);
    assert(logits != NULL);
    assert(logits->length == 2);

    printf("Logits produced\n");

    // Compute dL/dZ manually
    BASE_TYPE exp0 = exp(logits->data[0]);
    BASE_TYPE exp1 = exp(logits->data[1]);
    BASE_TYPE sum = exp0 + exp1;
    BASE_TYPE softmax0 = exp0 / sum;
    BASE_TYPE softmax1 = exp1 / sum;

    BASE_TYPE dL_dz0 = softmax0 - label->data[0];
    BASE_TYPE dL_dz1 = softmax1 - label->data[1];

    Vector *dLoss_dZ = create_vector(2);
    dLoss_dZ->data[0] = dL_dz0;
    dLoss_dZ->data[1] = dL_dz1;

    printf("dL/dZ calculated manually\n");

    // Compute activated output (softmax output) for .activated field
    Vector *activated = layer->activation.function.activation_loss_function.forward(logits);
    
    printf("activated calculated\n");

    BackpropContext context = {
        .type = dLoss_dActivation,
        .dLoss_dActivation = {
            .dL_dA = dLoss_dZ,
            .activated = activated
        }
    };

    printf("Context created\n");

    LayerGradients *grads = backward_layer(layer, input, logits, &context);
    assert(grads != NULL);
    assert(grads->d_weights != NULL);
    assert(grads->d_biases != NULL);
    assert(grads->d_inputs != NULL);

    printf("Backward layer\n");

    // Check gradients as before...

    printf("grads->d_biases[0]: %f, expected dL_dz0: %f\n", grads->d_biases->data[0], dL_dz0);
    assert(fabs(grads->d_biases->data[0] - dL_dz0) < EPSILON);
    assert(fabs(grads->d_biases->data[1] - dL_dz1) < EPSILON);

    size_t rows = grads->d_weights->rows;
    size_t cols = grads->d_weights->cols;
    assert(rows == 2 && cols == 3);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            BASE_TYPE expected = dLoss_dZ->data[i] * input->data[j];
            BASE_TYPE actual = grads->d_weights->data[i * cols + j];
            assert(fabs(actual - expected) < EPSILON);
        }
    }

    for (size_t i = 0; i < 3; i++) {
        BASE_TYPE expected =
            layer->weights->data[0 * 3 + i] * dL_dz0 +
            layer->weights->data[1 * 3 + i] * dL_dz1;
        assert(fabs(grads->d_inputs->data[i] - expected) < EPSILON);
    }

    BASE_TYPE old_weight = layer->weights->data[0];
    BASE_TYPE learning_rate = 0.01f;

    update_layer_parameters(layer, grads, learning_rate);
    BASE_TYPE new_weight = layer->weights->data[0];
    assert(fabs(new_weight - old_weight) > 0.0f);

    // Cleanup
    destroy_vector(dLoss_dZ);
    destroy_vector(activated);
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
