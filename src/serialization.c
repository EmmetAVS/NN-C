#include "serialization.h"
#include <stdio.h>
#include <stdlib.h>

#define SUCCESS true

static bool _write_layer(Layer *layer, FILE *file) {

    fwrite(&(layer->input_size), sizeof(size_t), 1, file);
    fwrite(&(layer->output_size), sizeof(size_t), 1, file);

    //Write weights
    for (size_t r = 0; r < layer->weights->rows; r ++) {

        for (size_t c = 0; c < layer->weights->cols; c ++) {

            BASE_TYPE value = matrix_get_value_at(layer->weights, r, c);
            fwrite(&(value), sizeof(BASE_TYPE), 1, file);

        }

    }

    //Write biases
    for (size_t i = 0; i < layer->biases->length; i ++) {

        BASE_TYPE value = layer->biases->data[i];
        fwrite(&(value), sizeof(BASE_TYPE), 1, file);

    }

    return SUCCESS;

}

static bool _read_layer(Layer *layer, FILE *file) {

    size_t input_size, output_size;
    fread(&input_size, sizeof(size_t), 1, file);
    fread(&output_size, sizeof(size_t), 1, file);

    if (input_size != layer->input_size || output_size != layer->output_size) {
        return !SUCCESS;
    }

    //Read weights
    for (size_t r = 0; r < layer->weights->rows; r ++) {

        for (size_t c = 0; c < layer->weights->cols; c ++) {

            BASE_TYPE value;
            fread(&(value), sizeof(BASE_TYPE), 1, file);
            matrix_set_value_at(layer->weights, r, c, value);

        }

    }

    //Read biases
    for (size_t i = 0; i < layer->biases->length; i ++) {

        BASE_TYPE value;
        fread(&(value), sizeof(BASE_TYPE), 1, file);
        layer->biases->data[i] = value;

    }

    return SUCCESS;

}

bool write_model_params(Model *model, const char *filename) {

    FILE *file = fopen(filename, "wb");
    if (!file) {
        return !SUCCESS;
    }

    fwrite(&(model->num_layers), sizeof(size_t), 1, file);
    
    for (size_t i = 0; i < model->num_layers; i ++) {

        Layer *layer = model->layers[i];
        if (!_write_layer(layer, file)) {
            return !SUCCESS;
        }

    }

    fclose(file);
    return SUCCESS;

}

bool load_model_params(Model *model, const char *filename) {

    FILE *file = fopen(filename, "rb");
    if (!file) {
        return !SUCCESS;
    }

    size_t layers;
    fread(&(layers), sizeof(size_t), 1, file);
    if (layers != model->num_layers) {
        return !SUCCESS;
    }
    
    for (size_t i = 0; i < model->num_layers; i ++) {

        Layer *layer = model->layers[i];
        if (!_read_layer(layer, file)) {
            return !SUCCESS;
        }

    }

    fclose(file);
    return SUCCESS;

}