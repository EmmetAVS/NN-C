#include "types.h"
#include <stdlib.h>

Vector *create_vector(size_t length) {

    Vector *v = (Vector *) malloc(sizeof(Vector));
    v->data = (BASE_TYPE *) calloc(length, sizeof(BASE_TYPE));
    v->length = length;

    return v;

}

void destroy_vector(Vector *vector) {

    free(vector->data);
    free(vector);

}

Matrix *create_matrix(size_t rows, size_t cols) {

    Matrix *m = (Matrix *) malloc(sizeof(Matrix));
    m->data = (BASE_TYPE *) calloc(rows * cols, sizeof(BASE_TYPE));
    m->rows = rows;
    m->cols = cols;

    return m;

}

void destroy_matrix(Matrix *matrix) {

    free(matrix->data);
    free(matrix);

}

BASE_TYPE matrix_get_value_at(Matrix *m, int rowIndex, int colIndex) {

    const int index = ((m->cols) * rowIndex) + colIndex;

    if (index >= m->cols * m-> rows) 
        return UNDEFINED;

    return m->data[index];

}

BASE_TYPE matrix_set_value_at(Matrix *m, int rowIndex, int colIndex, BASE_TYPE value) {

    const int index = ((m->cols) * rowIndex) + colIndex;

    if (index >= m->cols * m-> rows) 
        return UNDEFINED;

    const BASE_TYPE oldValue = m->data[index];

    m->data[index] = value;

    return oldValue;

}