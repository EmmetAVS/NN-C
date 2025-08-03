#include "model.h"
#include "optimizer.h"
#include "data.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

void load_train(CSVOutput *train, Vector ****inputs, Vector ****labels, size_t batches, size_t batch_size, size_t total_data_len, size_t pixel_count) {
    *inputs = (Vector ***)calloc(batches, sizeof(Vector **));
    *labels = (Vector ***)calloc(batches, sizeof(Vector **));

    for (size_t i = 0; i < batches; i ++) {

        (*inputs)[i] = (Vector **)calloc(batch_size, sizeof(Vector *));
        (*labels)[i] = (Vector **)calloc(batch_size, sizeof(Vector *));

    }

    for (size_t batch = 0; batch < batches; batch++) {

        for (size_t j = 0; j < batch_size; j ++) {
            size_t i = batch * batch_size + j;
            if (i >= total_data_len) break;

            Vector *curr = create_vector(pixel_count);

            for (size_t p = 0; p < pixel_count; p ++) {

                curr->data[p] = (float) (((int **) train->data)[i][p + 1]) / 255.0f;

            }

            const int label = (((int **) train->data)[i][0]);
            if (label > 9 || label < 0) {
                printf("Label: %d @ index %zu\n", label, i);
            }
            (*inputs)[batch][j] = curr;
            (*labels)[batch][j] = create_vector(10);
            (*labels)[batch][j]->data[label] = 1.f;

        }

    }
}

int main() {

    const char *data_path = "data/mnist.csv";
    const int epochs = 10;
    const int batch_size = 32;

    printf("Loading train dataset %s...\n", data_path);
    CSVOutput *train = read_csv(data_path, INTEGER);
    if (!train) {
        printf("Train dataset failed to load.\n");
        return 0;
    }

    Shuffler *shuffler = create_shuffler(train->data_rows);
    apply_shuffler(shuffler, train->data, sizeof(void*));
    destroy_shuffler(shuffler);
    
    const int pixel_count = train->cols - 1; // - 1 because of labels column

    const float test_percent = 0.2;
    int total_data_len = (int) (train->data_rows * (1 - test_percent));
    const int test_count = train->data_rows - total_data_len;
    const int batches = total_data_len / batch_size;

    
    //Setup model
    #define NUM_LAYERS 3

    size_t *shape[NUM_LAYERS] = {
        (size_t[2]) {pixel_count, 32},
        (size_t[2]) {32, 128},
        (size_t[2]) {128, 10},  
    };

    LossFunction empty_loss = {.backward = NULL, .forward = NULL};
    ActivationFunction activation_functions[NUM_LAYERS] = {activation_relu, activation_relu, activation_loss_softmax_cross_entropy};
    Model *model = create_model(shape, activation_functions, NUM_LAYERS, empty_loss);
    model_set_calculate_grads(model, true);

    Optimizer *opt = create_SGD_optimizer(0.01f);

    Vector ***inputs, ***labels;
    load_train(train, &inputs, &labels, batches, batch_size, total_data_len, pixel_count);

    Vector **test_inputs = (Vector **)malloc(sizeof(Vector *) * test_count);
    Vector **test_labels = (Vector **)malloc(sizeof(Vector *) * test_count);
    for (size_t i = total_data_len; i < train->data_rows; i ++) {

        size_t index = i - total_data_len;

        Vector *curr = create_vector(pixel_count);
        for (size_t p = 0; p < pixel_count; p ++) {

            curr->data[p] = (float) (((int **) train->data)[i][p + 1]) / 255.0f;

        }

        const int label = (((int **) train->data)[i][0]);
        test_inputs[index] = curr;
        test_labels[index] = create_vector(10);
        test_labels[index]->data[label] = 1.f;

    }
    
    
    printf("Training with %d epochs, %d batch size, and %d batches across %d examples\n", epochs, batch_size, batches, total_data_len);

    
    for (int epoch = 0; epoch < epochs; ++epoch) {

        BASE_TYPE total_loss = 0;
        size_t data_used = 0;

        for (int i = 0; i < batches; ++i) {
            
            model_zero_grads(model);
            model_set_max_grads(model, batch_size);

            for (int b = 0; b < batch_size; b ++) {
                if (inputs[i][b] && labels[i][b]) {

                    Vector *output = model_forward(model, inputs[i][b]);
                    total_loss += cross_entropy_loss(output, labels[i][b]);
                    data_used += 1;
                    model_backward(model, labels[i][b]);
                    destroy_vector(output);

                }
            }

            model_average_grads(model);
            model_step(model, opt);

        }
        
        BASE_TYPE averaged_loss = total_loss / (data_used);
        printf("Loss: %f @ epoch #%d\n", averaged_loss, epoch + 1);
    }

    //Test model

    model_set_calculate_grads(model, false);

    int successes = 0;
    BASE_TYPE total_loss = 0;

    for (size_t i = 0; i < test_count; i ++) {

        if (!test_inputs[i] || !test_labels[i]) continue;
        Vector *output = model_forward(model, test_inputs[i]);
        size_t index = argmax(output);
        size_t label_index = argmax(test_labels[i]);
        BASE_TYPE loss = cross_entropy_loss(output, test_labels[i]);

        if (label_index == index) {
            successes += 1;
        }

        total_loss += loss;
        destroy_vector(output);

    }

    BASE_TYPE averaged_loss = total_loss / test_count;

    printf("Testing: \nAveraged Loss: %f\nSuccesses: %d/%d\n", averaged_loss, successes, test_count);
    /*
    should output to ~
    Testing:
    Averaged Loss: 0.249387
    Successes: 7790/8400
    */

    //Cleanup
    for (size_t batch = 0; batch < batches; batch++) {

        for (size_t j = 0; j < batch_size; j ++) {
            size_t i = batch * batch_size + j;
            if (i >= total_data_len) break;

            destroy_vector(inputs[batch][j]);
            destroy_vector(labels[batch][j]);

        }

        free(inputs[batch]);
        free(labels[batch]);
    }

    free(inputs);
    free(labels);

    for (size_t i = 0; i < test_count; i ++) {
        destroy_vector(test_inputs[i]);
        destroy_vector(test_labels[i]);
    }
    free(test_inputs);
    free(test_labels);
    
    destroy_model(model);
    destroy_optimizer(opt);
    destroy_csv_output(train);
    return 0;

}