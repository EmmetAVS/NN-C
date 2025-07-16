#include "loss.h"
#include "types.h"
#include <math.h>

static BASE_TYPE mean_squared_error_loss_forward(Vector *outputs, Vector *labels) {

    if (outputs->length != labels->length) {
        return UNDEFINED;
    }

    BASE_TYPE sum = 0;

    for (size_t i = 0; i < outputs->length; i ++) {

        BASE_TYPE diff = outputs->data[i] - labels->data[i];

        sum += (diff * diff);

    }

    return sum / (outputs->length);

}

static Vector *mean_squared_error_loss_backward(Vector *outputs, Vector *labels) {
    
    Vector *grad = create_vector(outputs->length);

    for (size_t i = 0; i < grad->length; i ++) {

        grad->data[i] = 2 * (outputs->data[i] - labels->data[i]);

    }

    return grad;

}

LossFunction mean_squared_error_loss = {
    .forward = mean_squared_error_loss_forward,
    .backward = mean_squared_error_loss_backward
};

static BASE_TYPE cross_entropy_loss_forward(Vector *outputs, Vector *labels) {
    
    if (outputs->length != labels->length) {
        return UNDEFINED;
    }

    BASE_TYPE sum = 0;

    for (size_t i = 0; i < outputs->length; i ++) {

        sum += (labels->data[i]) * (log(outputs->data[i] + EPSILON));

    }

    return sum * (-1);

}

LossFunctionForward cross_entropy_loss = cross_entropy_loss_forward;

BASE_TYPE calculate_cost(Vector **outputs_list, Vector **labels_list, size_t batch_size, LossFunctionForward loss_function) {

    BASE_TYPE sum = 0;

    for (size_t i = 0; i < batch_size; i ++) {

        BASE_TYPE result = loss_function(outputs_list[i], labels_list[i]);

        if (isnan(result)) {
            return UNDEFINED;
        }

        sum += result;

    }

    return (sum / batch_size);

}