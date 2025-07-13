#include "loss.h"
#include <math.h>

static const BASE_TYPE epsilon = 1e-9; //Prevent NaN's from log(0)

BASE_TYPE mean_squared_error_loss(Vector *outputs, Vector *expected) {

    if (outputs->length != expected->length) {
        return UNDEFINED;
    }

    BASE_TYPE sum = 0;

    for (int i = 0; i < outputs->length; i ++) {

        BASE_TYPE diff = outputs->data[i] - expected->data[i];

        sum += (diff * diff);

    }

    return sum / (outputs->length);

}

BASE_TYPE cross_entropy_loss(Vector *outputs, Vector *expected) {
    
    if (outputs->length != expected->length) {
        return UNDEFINED;
    }

    BASE_TYPE sum = 0;

    for (int i = 0; i < outputs->length; i ++) {

        sum += (expected->data[i]) * (log(outputs->data[i] + epsilon));

    }

    return sum * (-1);

}

BASE_TYPE calculate_cost(Vector **outputs, Vector **expected, size_t batch_size, LossFunction loss_function) {

    BASE_TYPE sum = 0;

    for (int i = 0; i < batch_size; i ++) {

        BASE_TYPE result = loss_function(outputs[i], expected[i]);

        if (isnan(result)) {
            return UNDEFINED;
        }

        sum += result;

    }

    return (sum / batch_size);

}