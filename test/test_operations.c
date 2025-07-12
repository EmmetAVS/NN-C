#include "operations.h"
#include "types.h"
#include <assert.h>

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

    Vector *v2 = create_vector(size + 1);
    Matrix *m2 = create_matrix(size + 1, size);
    
    assert(multiply_matrix_with_vector(m, v2) == NULL);
    assert(multiply_matrix_with_vector(m2, v) == NULL);

    v2 = create_vector(size);

    for (int i = 0; i < v2->length; i ++) {

        v2->data[i] = (-1) * i;

    }

    Vector *sum = add_vector_to_vector(v, v2);
    for (int i = 0; i < sum->length; i ++) {

        assert(sum->data[i] == 0.f);

    }

    return 0;

}