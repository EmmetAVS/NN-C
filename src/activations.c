#include "activations.h"
#include "types.h"
#include "loss.h"
#include <math.h>

#define E_CONSTANT 2.71828182845904523536

static Vector *relu_forward(Vector *logits) {

    Vector *new = create_vector(logits->length);

    for (size_t i = 0; i < logits->length; i ++) {

        BASE_TYPE cur_value = logits->data[i];
        new->data[i] = cur_value > EPSILON ? cur_value : 0;

    }

    return new;

}

static Vector *relu_backward(Vector *logits, Vector *activated) {

    Vector *new = create_vector(logits->length);

    for (size_t i = 0; i < logits->length; i ++) {

        BASE_TYPE cur_value = logits->data[i];
        new->data[i] = cur_value > EPSILON ? 1 : 0;

    }

    return new;

}

const RawActivationFunction function_activation_relu = {
    .forward = relu_forward,
    .backward = relu_backward
};

static Vector *sigmoid_forward(Vector *logits) {

    Vector *new = create_vector(logits->length);

    for (size_t i = 0; i < logits->length; i ++) {

        BASE_TYPE cur_value = logits->data[i];
        new->data[i] = (1 / (1 + pow(E_CONSTANT, (-1) * cur_value))); // 1/(1 + e^(-x))

    }

    return new;

}

static Vector *sigmoid_backward(Vector *logits, Vector *activated) {

    Vector *new = create_vector(logits->length);

    for (size_t i = 0; i < logits->length; i ++) {

        BASE_TYPE cur_value = activated->data[i];
        new->data[i] = (cur_value) * (1 - cur_value);

    }

    return new;

}

const RawActivationFunction function_activation_sigmoid = {
    .forward = sigmoid_forward,
    .backward = sigmoid_backward
};

static Vector *softmax_forward(Vector *logits) {

    Vector *new = create_vector(logits->length);

    BASE_TYPE sum = 0;
    BASE_TYPE max_value = NAN;

    for (size_t i = 0; i < logits->length; i ++) {

        if (isnan(max_value) || logits->data[i] > max_value) {

            max_value = logits->data[i];

        }

    }

    for (size_t i = 0; i < logits->length; i ++) {

        sum += pow(E_CONSTANT, logits->data[i] - max_value);

    }

    for (size_t i = 0; i < logits->length; i ++) {

        new->data[i] = (pow(E_CONSTANT, logits->data[i] - max_value)) / (sum);

    }

    return new;

}

static BASE_TYPE softmax_with_cross_entropy_loss_forward(Vector *logits, Vector *labels) {
    
    Vector *softmaxed = softmax_forward(logits);

    BASE_TYPE loss = cross_entropy_loss(softmaxed, labels);

    destroy_vector(softmaxed);

    return loss;

}

static Vector *softmax_with_cross_entropy_loss_backward(Vector *logits, Vector *labels) {

    Vector *new = softmax_forward(logits);

    for (size_t i = 0; i < new->length; i ++) {

        new->data[i] -= labels->data[i];

    }

    return new;

}

const ActivationLossFunction function_activation_loss_softmax_cross_entropy = {
    .forward = softmax_forward,
    .forward_with_loss = softmax_with_cross_entropy_loss_forward,
    .backward = softmax_with_cross_entropy_loss_backward
};

ActivationFunction activation_relu = {
    .type = RAW,
    .function.activation_function = function_activation_relu
};

ActivationFunction activation_sigmoid = {
    .type = RAW,
    .function.activation_function = function_activation_sigmoid
};

ActivationFunction activation_loss_softmax_cross_entropy = {
    .type = WITH_LOSS,
    .function.activation_loss_function = function_activation_loss_softmax_cross_entropy
};

