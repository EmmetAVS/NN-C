#include "types.h"
#include "tests.h"
#include "tensor.h"

int main() {

    Tensor *tensor = create_tensor(3, 4, 2, 6);
    BASE_TYPE *val = tensor_access_value_at(tensor, 1, 1, 1);
    *val = 3;
    assert(*tensor_access_value_at(tensor, 1, 1, 1) == 3);

    val = tensor_access_value_at(tensor, 3, 1, 5);
    *val = 4;
    assert(*tensor_access_value_at(tensor, 3, 1, 5) == 4);

    size_t indexes[3] = {(size_t) 3, (size_t) 1, (size_t) 5};
    val = tensor_access_value_at_from_indexes(tensor, indexes);
    assert(*val == 4);

    destroy_tensor(tensor);

    return 0;

}