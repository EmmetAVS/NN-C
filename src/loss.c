#include "loss.h"
#include "types.h"
#include <math.h>

static BASE_TYPE mean_squared_error_loss_forward(Vector *inputs, Vector *labels) {

    if (inputs->length != labels->length) {
        return UNDEFINED;
    }

    BASE_TYPE sum = 0;

    for (size_t i = 0; i < inputs->length; i ++) {

        BASE_TYPE diff = inputs->data[i] - labels->data[i];

        sum += (diff * diff);

    }

    return sum / (inputs->length);

}

static Vector *mean_squared_error_loss_backward(Vector *inputs, Vector *labels) {
    
    Vector *grad = create_vector(inputs->length);

    for (size_t i = 0; i < grad->length; i ++) {

        grad->data[i] = 2 * (inputs->data[i] - labels->data[i]);

    }

    return grad;

}

LossFunction mean_squared_error_loss = {
    .forward = mean_squared_error_loss_forward,
    .backward = mean_squared_error_loss_backward
};

static BASE_TYPE cross_entropy_loss_forward(Vector *inputs, Vector *labels) {
    
    if (inputs->length != labels->length) {
        return UNDEFINED;
    }

    BASE_TYPE sum = 0;

    for (size_t i = 0; i < inputs->length; i ++) {

        sum += (labels->data[i]) * (log(inputs->data[i] + EPSILON));

    }

    return sum * (-1);

}

LossFunctionForward cross_entropy_loss = cross_entropy_loss_forward;

BASE_TYPE calculate_cost(Vector **inputs_list, Vector **labels_list, size_t batch_size, LossFunctionForward loss_function) {

    BASE_TYPE sum = 0;

    for (size_t i = 0; i < batch_size; i ++) {

        BASE_TYPE result = loss_function(inputs_list[i], labels_list[i]);

        if (isnan(result)) {
            return UNDEFINED;
        }

        sum += result;

    }

    return (sum / batch_size);

}