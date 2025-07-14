#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "types.h"

Vector *add_vector_to_vector(Vector *v1, Vector *v2);
Vector *multiply_matrix_with_vector(Matrix *m, Vector *v);
Vector *average_vectors(Vector **vectors, size_t length);
Vector *multiply_vector_contents(Vector *v1, Vector *v2);

#endif