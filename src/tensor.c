#include "types.h"
#include "tensor.h"
#include <stdlib.h>
#include <stdarg.h>

Tensor *create_tensor(size_t dimensions, ...) {
    
    size_t *shape = (size_t *)malloc(sizeof(size_t) * dimensions);

    va_list args;
    va_start(args, dimensions);
    for (size_t i = 0; i < dimensions; i ++) {
        shape[i] = va_arg(args, size_t);
    }

    va_end(args);
    return create_tensor_from_shape(dimensions, shape);

}

Tensor *create_tensor_from_shape(size_t dimensions, size_t *shape) {

    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    tensor->dimensions = dimensions;
    tensor->shape = shape;
    tensor->strides = (size_t *)malloc(sizeof(size_t) * dimensions);
    
    size_t total_size = 1;

    for (size_t i = dimensions; i-- > 0;) {

        tensor->strides[i] = total_size;
        total_size *= tensor->shape[i];

    }

    tensor->data = (BASE_TYPE *)calloc(total_size, sizeof(BASE_TYPE));
    return tensor;

}

Tensor *duplicate_tensor(Tensor *tensor) {

    size_t *shape = (size_t *)malloc(sizeof(size_t) * tensor->dimensions);
    size_t total_size = 1;
    
    for (size_t i = 0; i < tensor->dimensions; i ++) {
        shape[i] = tensor->shape[i];
        total_size *= tensor->shape[i];
    }

    Tensor *new_tensor = create_tensor_from_shape(tensor->dimensions, shape);
    
    for (size_t i = 0; i < total_size; i ++) {
        new_tensor->data[i] = tensor->data[i];
    }

    return new_tensor;

}

void destroy_tensor(Tensor *tensor) {

    free(tensor->data);
    free(tensor->strides);
    free(tensor->shape);
    free(tensor);

}

Tensor *add_tensors(Tensor *t1, Tensor *t2) {

    Tensor *new = duplicate_tensor(t1);

    size_t total_size = tensor_calculate_total_size(t1);
    
    for (size_t i = 0; i < total_size; i ++) {
        new->data[i] += t2->data[i];
    }

    return new;

}

Tensor *tensor_batched_matrix_multiply(Tensor *t1, Tensor *t2) {

    size_t *shape = (size_t *)malloc(sizeof(size_t) * t1->dimensions);

    for (size_t i = 0; i < t1->dimensions; i ++) {

        shape[i] = t1->shape[i];
    }

    shape[t1->dimensions - 1] = t2->shape[t1->dimensions - 1];

    Tensor *new = create_tensor_from_shape(t1->dimensions, shape);

    size_t *indexes = (size_t *)malloc(sizeof(size_t) * t1->dimensions);
    tensor_batched_matrix_multiply_dimension_indexed(t1, t2, new, 0, indexes);
    free(indexes);

    return new;

}

/*
Indexes must be of length t1->dimensions = t2->dimensions
*/
static void tensor_batched_matrix_multiply_dimension_indexed(Tensor *t1, Tensor *t2, 
    Tensor *new, size_t dim, size_t *indexes) {

    if (dim + 2 == t1->dimensions) {

        for (size_t r = 0; r < t1->shape[dim]; r ++) {
            for (size_t c = 0; c < t2->shape[dim + 1]; c ++) {

                indexes[dim] = r;
                indexes[dim + 1] = c;

                BASE_TYPE *value = tensor_access_value_at_from_indexes(new, indexes);
                for (size_t i = 0; i < t1->shape[dim + 1]; i ++) {

                    indexes[dim + 1] = i;
                    BASE_TYPE v1 = tensor_access_raw_value_at_from_indexes(t1, indexes);

                    indexes[dim + 1] = c;
                    indexes[dim] = i;

                    BASE_TYPE v2 = tensor_access_raw_value_at_from_indexes(t2, indexes);
                    *value += v1 * v2;

                }

            }
        }

        return;

    }

    for (size_t i = 0; i < t1->shape[dim]; i ++) {

        indexes[dim] = i;
        tensor_batched_matrix_multiply_dimension_indexed(t1, t2, new, dim + 1, indexes);
    }

}