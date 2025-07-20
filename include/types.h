#ifndef TYPES_H
#define TYPES_H

#include <math.h>
#include <stddef.h>
#include <stdbool.h>

#define BASE_TYPE float
#define UNDEFINED NAN
#define EPSILON 1e-9

typedef struct Vector {

    BASE_TYPE *data;
    size_t length;

} Vector;

typedef struct Matrix {
    BASE_TYPE *data; //Treated as 2 Dimensional
    size_t rows;
    size_t cols;
} Matrix;

Vector *create_vector(size_t length);
void destroy_vector(Vector *vector);
Matrix *create_matrix(size_t rows, size_t cols);
void destroy_matrix(Matrix *matrix);
BASE_TYPE matrix_get_value_at(Matrix *m, int rowIndex, int colIndex);
BASE_TYPE matrix_set_value_at(Matrix *m, int rowIndex, int colIndex, BASE_TYPE value);

#endif