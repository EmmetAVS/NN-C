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

    for (size_t i = dimensions; i --> 0;) {

        tensor->strides[i] = total_size;
        total_size *= tensor->shape[i];

    }

    tensor->data = (BASE_TYPE *)calloc(total_size, sizeof(BASE_TYPE));
    return tensor;

}

void destroy_tensor(Tensor *tensor) {

    free(tensor->data);
    free(tensor->strides);
    free(tensor->shape);
    free(tensor);

}