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

    if (m->cols != v->length || m->rows != m->cols)
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