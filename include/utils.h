#ifndef UTILS_H
#define UTILS_H

#include "types.h"

typedef struct Shuffler {

    size_t length;
    size_t *new_indexes;

} Shuffler;

Vector *one_hot_encode(Vector *vec);
size_t argmax(Vector *vec);
Shuffler *create_shuffler(size_t length);
void destroy_shuffler(Shuffler *shuffler);
void apply_shuffler(Shuffler *shuffler, void *data, size_t element_size);

#endif