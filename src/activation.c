#include "activation.h"
#include "types.h"

#define E_CONSTANT 2.71828182845904523536

Vector* relu(Vector* input) {

    Vector *new = create_vector(input->length);

    for (int i = 0; i < input->length; i ++) {

        BASE_TYPE cur_value = input->data[i];
        new->data[i] = cur_value > EPSILON ? cur_value : 0;

    }

    return new;

}

Vector* sigmoid(Vector* input) {

    Vector *new = create_vector(input->length);

    for (int i = 0; i < input->length; i ++) {

        BASE_TYPE cur_value = input->data[i];
        new->data[i] = (1 / (1 + pow(E_CONSTANT, (-1) * cur_value))); // 1/(1 + e^(-x))

    }

    return new;

}

Vector* flatten(Matrix* input) {

    Vector* new = create_vector(input->rows * input->cols);

    for (int i = 0; i < new->length; i ++) {

        new->data[i] = input->data[i];

    }

    return new;

}
