#include "operations.h"
#include "types.h"
#include "tests.h"

int main() {
    
    nnlib_startup();

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

        CHECK(product->data[i] == values[i]);

    }
    destroy_vector(product);

    Vector *v2 = create_vector(size + 1);
    
    CHECK(multiply_matrix_with_vector(m, v2) == NULL);
    destroy_vector(v2);

    v2 = create_vector(size);

    for (int i = 0; i < v2->length; i ++) {

        v2->data[i] = (-1) * i;

    }

    Vector *sum = add_vector_to_vector(v, v2);
    for (int i = 0; i < sum->length; i ++) {

        CHECK(sum->data[i] == 0.f);

    }


    Matrix *m2 = create_matrix(size, size + 1);

    for (int r = 0; r < m2->rows; r ++) {
        for (int c = 0; c < m2-> cols; c ++ ) {
            matrix_set_value_at(m2, r, c, sin(r) + cos(c));
        }
    }

    Matrix *transposed = transpose_matrix(m2);

    for (int r = 0; r < m->rows; r ++) {
        for (int c = 0; c < m-> cols; c ++ ) {
            BASE_TYPE transposed_value = matrix_get_value_at(transposed, r, c);
            BASE_TYPE original_value = matrix_get_value_at(m2, c, r);
            CHECK(FLOAT_EQ(transposed_value, original_value));
        }
    }

    destroy_vector(v);
    destroy_matrix(m);
    destroy_vector(v2);
    destroy_vector(sum);
    destroy_matrix(m2);
    destroy_matrix(transposed);

    return 0;

}