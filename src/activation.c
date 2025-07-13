#include "activation.h"
#include "types.h"
#include <math.h>

#define E_CONSTANT 2.71828182845904523536

Vector* relu(Vector* input) {

    Vector *new = create_vector(input->length);

    for (size_t i = 0; i < input->length; i ++) {

        BASE_TYPE cur_value = input->data[i];
        new->data[i] = cur_value > EPSILON ? cur_value : 0;

    }

    return new;

}

Vector* sigmoid(Vector* input) {

    Vector *new = create_vector(input->length);

    for (size_t i = 0; i < input->length; i ++) {

        BASE_TYPE cur_value = input->data[i];
        new->data[i] = (1 / (1 + pow(E_CONSTANT, (-1) * cur_value))); // 1/(1 + e^(-x))

    }

    return new;

}

Vector* softmax(Vector* input) {

    Vector *new = create_vector(input->length);

    BASE_TYPE sum = 0;
    BASE_TYPE max_value = NAN;

    for (size_t i = 0; i < input->length; i ++) {

        if (isnan(max_value) || input->data[i] > max_value) {

            max_value = input->data[i];

        }

    }

    for (size_t i = 0; i < input->length; i ++) {

        sum += pow(E_CONSTANT, input->data[i] - max_value);

    }

    for (size_t i = 0; i < input->length; i ++) {

        new->data[i] = (pow(E_CONSTANT, input->data[i] - max_value)) / (sum);

    }

}

Vector* flatten(Matrix* input) {

    Vector* new = create_vector(input->rows * input->cols);

    for (size_t i = 0; i < new->length; i ++) {

        new->data[i] = input->data[i];

    }

    return new;

}
