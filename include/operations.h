#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "types.h"

Vector *add_vector_to_vector(Vector *v1, Vector *v2);
Vector *multiply_matrix_with_vector(Matrix *m, Vector *v);
Vector *average_vectors(Vector **vectors, size_t length);
Vector *multiply_vector_contents(Vector *v1, Vector *v2);
Vector* flatten(Matrix* input);
Matrix *average_matrices(Matrix **matrices, size_t length);
Matrix *transpose_matrix(Matrix *m);
Matrix *multiply_matrices(Matrix *m1, Matrix *m2);

static inline void scalar_multiply_matrix(Matrix *m, BASE_TYPE scalar) {

    for (size_t r = 0; r < m->rows; r ++) {
        for (size_t c = 0; c < m->cols; c ++) {
            m->data[MATRIX_INDEX(m, r, c)] *= scalar;
        }
    }

}

#endif