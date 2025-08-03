#include "utils.h"
#include <stdlib.h>
#include <string.h>

Vector *one_hot_encode(Vector *vec) {

    if (!vec) return NULL;

    Vector *encoded = create_vector(vec->length);

    size_t max_index = 0;
    for (size_t i = 0; i < vec->length; i ++) {

        if (vec->data[i] > vec->data[max_index]) {

            max_index = i;

        }

    }

    encoded->data[max_index] = 1.f;
    return encoded;

}

size_t argmax(Vector *vec) {

    size_t max_index = 0;
    for (size_t i = 0; i < vec->length; i ++) {

        if (vec->data[i] > vec->data[max_index]) {

            max_index = i;

        }

    }

    return max_index;

}

Shuffler *create_shuffler(size_t length) {

    Shuffler *shuffler = (Shuffler *)malloc(sizeof(Shuffler));
    shuffler->length = length;
    shuffler->new_indexes = (size_t *)malloc(sizeof(size_t) * length);

    HANDLE_SRAND()
    for (size_t i = 0; i < length; i ++) {

        (shuffler->new_indexes)[i] = i;

    }

    for (size_t i = 0; i < length; i ++) {

        (shuffler->new_indexes)[i] = i;

    }

    for (size_t i = length - 1; i > 0; i--) {
        size_t j = rand() % (i + 1);
        size_t temp = shuffler->new_indexes[i];
        shuffler->new_indexes[i] = shuffler->new_indexes[j];
        shuffler->new_indexes[j] = temp;
    }

    return shuffler;

}

void destroy_shuffler(Shuffler *shuffler) {

    if (!shuffler) return;
    if (shuffler->new_indexes) {
        free(shuffler->new_indexes);
    }

    free(shuffler);

}

static void _swap_elements(void *data, size_t element_size, size_t index_one, size_t index_two) {

    if (index_one == index_two) return;
    char *bytes = (char *)data;
    void *index_one_ptr = (void *)(bytes + ((size_t) element_size * index_one));
    void *index_two_ptr = (void *)(bytes + ((size_t) element_size * index_two));
    void *original_data = malloc(element_size);
    memcpy(original_data, index_one_ptr, element_size);

    memcpy(index_one_ptr, index_two_ptr, element_size);
    memcpy(index_two_ptr, original_data, element_size);
    free(original_data);

}

void apply_shuffler(Shuffler *shuffler, void *data, size_t element_size) {

    for (size_t i = 0; i < shuffler->length; i ++) {

        size_t new_index = shuffler->new_indexes[i];
        _swap_elements(data, element_size, i, new_index);

    }

}