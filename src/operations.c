#include "operations.h"
#include "types.h"

Vector *add_vector_to_vector(Vector *v1, Vector *v2) {

    if (v1->length != v2->length)
        return NULL;

    const int length = v1->length;

    Vector *new = create_vector(length);

    for (size_t i = 0; i < length; i ++) {

        new->data[i] = v1->data[i] + v2->data[i];

    }

    return new;

}

Vector *multiply_matrix_with_vector(Matrix *m, Vector *v) {

    if (m->cols != v->length)
        return NULL;

    Vector *new = create_vector(m->cols);

    for (size_t i = 0; i < m->rows; i ++) {

        BASE_TYPE sum = 0;

        for (size_t c = 0; c < m->cols; c++) {
            sum += matrix_get_value_at(m, i, c) * v->data[c];
        }

        new->data[i] = sum;

    }

    return new;

}

Vector *average_vectors(Vector **vectors, size_t length) {

    Vector *avg = create_vector((vectors[0])->length);

    for (size_t i = 0; i < length; i ++) {

        for (size_t j = 0; j < avg->length; j ++) {

            avg->data[j] += vectors[i]->data[j];

        }

    }

    for (size_t i = 0; i < avg->length; i ++) {

        avg->data[i] /= (BASE_TYPE) length;

    }

    return avg;

}

Vector *multiply_vector_contents(Vector *v1, Vector *v2) {

    Vector *prod = create_vector(v1->length);

    for (size_t i = 0; i < prod->length; i ++) {

        prod->data[i] = (v1->data[i]) * (v2->data[i]);

    }

    return prod;

}

Vector *flatten(Matrix *input) {

    Vector* new = create_vector(input->rows * input->cols);

    for (size_t i = 0; i < new->length; i ++) {

        new->data[i] = input->data[i];

    }

    return new;

}
