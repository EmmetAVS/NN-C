#include "types.h"
#include "tests.h"
#include <stdio.h>

int main() {
    
    nnlib_startup();

    //Test Vector Instantiation/Destruction
    Vector *v = create_vector(10);
    destroy_vector(v);

    //Test Matrix Instantiation/Destruction
    Matrix *m = create_matrix(10, 10);
    destroy_matrix(m);

    //Test Matrix Indexing
    m = create_matrix(10, 10);
    CHECK(matrix_get_value_at(m, 0, 0) == 0.f);
    CHECK(matrix_set_value_at(m, 0, 0, 1) == 0.f);
    CHECK(matrix_get_value_at(m, 0, 0) == 1.f);
    CHECK(isnan(matrix_get_value_at(m, 10, 9)));
    CHECK(isnan(matrix_get_value_at(m, 9, 10)));

    destroy_matrix(m);

    return 0;

}