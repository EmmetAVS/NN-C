#include "types.h"
#include <stdarg.h>
#include <stdlib.h>

typedef struct Tensor {
    BASE_TYPE *data;
    size_t *shape;
    size_t *strides;
    size_t dimensions;
} Tensor;

Tensor *create_tensor(size_t dimensions, ...);
Tensor *create_tensor_from_shape(size_t dimensions, size_t *shape);
Tensor *duplicate_tensor(Tensor *tensor);
void destroy_tensor(Tensor *tensor);

Tensor *add_tensors(Tensor *t1, Tensor *t2);
Tensor *tensor_batched_matrix_multiply(Tensor *t1, Tensor *t2);

static inline size_t tensor_calculate_total_size(Tensor *tensor) {

    size_t total_size = 1;

    for (size_t i = 0; i < tensor->dimensions; i ++) {
        total_size *= tensor->shape[i];
    }

    return total_size;
}

static inline BASE_TYPE *tensor_access_value_at_from_indexes(Tensor *tensor, size_t *indexes) {

    size_t index = 0;

    for (size_t i = 0; i < tensor->dimensions; i ++) {

        index += tensor->strides[i] * indexes[i];

    }

    return &(tensor->data[index]);

}


static inline BASE_TYPE tensor_access_raw_value_at_from_indexes(Tensor *tensor, size_t *indexes) {

    size_t index = 0;

    for (size_t i = 0; i < tensor->dimensions; i ++) {

        index += tensor->strides[i] * indexes[i];

    }

    return tensor->data[index];

}

static inline BASE_TYPE *tensor_access_value_at(Tensor *tensor, ...) {

    size_t *indexes = (size_t *)malloc(sizeof(size_t) * tensor->dimensions);

    va_list args;
    va_start(args, tensor);
    for (size_t i = 0; i < tensor->dimensions; i ++) {
        indexes[i] = va_arg(args, size_t);
    }

    va_end(args);
    
    BASE_TYPE *index = tensor_access_value_at_from_indexes(tensor, indexes);
    free(indexes);
    return index;

}