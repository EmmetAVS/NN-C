#include "operations.h"
#include "types.h"
#include <assert.h>

/*

To Test

Vector *add_vector_to_vector(Vector *v1, Vector *v2);
Vector *multiply_matrix_with_vector(Matrix *m, Vector *v);
Vector *average_vectors(Vector **vectors, size_t length);
Vector *multiply_vector_contents(Vector *v1, Vector *v2);
Vector* flatten(Matrix* input);
Matrix *average_matrices(Matrix **matrices, size_t length);

*/

int main() {

    const int size = 3;
    const BASE_TYPE values[] = {5.f, 8.f, 11.f}; 

    Vector *v = create_vector(size);

    for (int i = 0; i < v->length; i ++) {
        v->data[i] = i;
    }

    Matrix *m = create_matrix(size, size);

    for (int r = 0; r < m->rows; r ++) {

        for (int c = 0; c < m-> cols; c ++ ) {

            matrix_set_value_at(m, r, c, r + c);

        }

    }

    Vector *product = multiply_matrix_with_vector(m, v);

    for (int i = 0; i < product -> length; i ++) {

        assert(product->data[i] == values[i]);

    }
    destroy_vector(product);

    Vector *v2 = create_vector(size + 1);
    
    assert(multiply_matrix_with_vector(m, v2) == NULL);
    destroy_vector(v2);

    v2 = create_vector(size);

    for (int i = 0; i < v2->length; i ++) {

        v2->data[i] = (-1) * i;

    }

    Vector *sum = add_vector_to_vector(v, v2);
    for (int i = 0; i < sum->length; i ++) {

        assert(sum->data[i] == 0.f);

    }

    destroy_vector(v);
    destroy_matrix(m);
    destroy_vector(v2);
    destroy_vector(sum);

    return 0;

}