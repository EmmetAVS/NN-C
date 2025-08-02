#include "utils.h"

Vector *one_hot_encode(Vector *vec) {

    if (!vec) return NULL;

    Vector *encoded = create_vector(vec->length);

    size_t max_index = 0;
    for (size_t i = 0; i < vec->length; i ++) {

        if (vec->data[i] > vec->data[max_index]) {

            max_index = i;

        }

    }

    encoded->data[max_index] = 1.f;
    return encoded;

}

size_t argmax(Vector *vec) {

    size_t max_index = 0;
    for (size_t i = 0; i < vec->length; i ++) {

        if (vec->data[i] > vec->data[max_index]) {

            max_index = i;

        }

    }

    return max_index;

}