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

    for (size_t i = 0; i < 3 * 2; i++) {
        layer->weights->data[i] = 0.1f;
    }
    for (size_t i = 0; i < 2; i++) {
        layer->biases->data[i] = 0.0f;
    }

    Vector *activated = forward_layer(layer, input, true);
    assert(activated != NULL);
    assert(activated->length == 2);

    Vector *logits = create_vector(2);
    for (size_t i = 0; i < 2; i++) {
        BASE_TYPE sum = 0.0f;
        for (size_t j = 0; j < 3; j++) {
            sum += matrix_get_value_at(layer->weights, i, j) * input->data[j];
        }
        sum += layer->biases->data[i];
        logits->data[i] = sum;
    }

    BASE_TYPE exp0 = exp(logits->data[0]);
    BASE_TYPE exp1 = exp(logits->data[1]);
    BASE_TYPE sum_exp = exp0 + exp1;
    BASE_TYPE softmax0 = exp0 / sum_exp;
    BASE_TYPE softmax1 = exp1 / sum_exp;

    BASE_TYPE dL_dz0 = softmax0 - label->data[0];
    BASE_TYPE dL_dz1 = softmax1 - label->data[1];

    Vector *dLoss_dZ = create_vector(2);
    dLoss_dZ->data[0] = dL_dz0;
    dLoss_dZ->data[1] = dL_dz1;

    BackpropContext context = {
        .type = LabelsOutput,
        .labels_output = {
            .labels = label
        }
    };

    LayerGradients *grads = backward_layer(layer, input, logits, &context);
    assert(grads != NULL);
    assert(grads->d_weights != NULL);
    assert(grads->d_biases != NULL);
    assert(grads->d_inputs != NULL);

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
            matrix_get_value_at(layer->weights, 0, i) * dL_dz0 +
            matrix_get_value_at(layer->weights, 1, i) * dL_dz1;
        assert(fabs(grads->d_inputs->data[i] - expected) < EPSILON);
    }

    BASE_TYPE old_weight = layer->weights->data[0];
    BASE_TYPE learning_rate = 0.01f;

    update_layer_parameters(layer, grads, learning_rate);
    BASE_TYPE new_weight = layer->weights->data[0];
    assert(fabs(new_weight - old_weight) > 0.0f);

    destroy_vector(dLoss_dZ);
    destroy_vector(logits);
    destroy_vector(activated);
    destroy_vector(input);
    destroy_vector(label);
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
