#include "types.h"
#include <stdlib.h>
#include <time.h>
#include "activations.h"

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

unsigned int _time() {
    return (unsigned int) time(NULL);
}

State _state = {
    .srand_called = false,
    .srand_seed_set = false
};

bool nnlib_startup() {

    _init_activations();

    return true;

}