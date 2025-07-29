#include "model.h"
#include "optimizer.h"
#include "data.h"
#include <stdio.h>

int main() {

    const int epochs = 2;
    const int batch_size = 128;

    CSVOutput *train = read_csv("data/mnist_train.csv", INTEGER);
    if (!train) {
        printf("Train dataset failed to load.\n");
        return 0;
    }
    
    const int pixel_count = train->cols - 1; // - 1 because of labels column

    int total_data_len = (int) train->data_rows;
    const int batches = total_data_len / batch_size;

    /*
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

    Optimizer *o = create_SGD_optimizer(1.f);

    Vector *inputs[batches][batch_size], *labels[batches][batch_size];
    for (size_t batch = 0; batch < batches; batch++) {

        for (size_t i = 0; i < total_data_len, i < (batch_size * (i + 1)); i ++) {
            Vector *curr = create_vector(pixel_count);

            for (size_t p = 0; p < pixel_count; p ++) {

                curr->data[p] = (float) (((int **) train->data)[i][p + 1]);

            }

            const int label = (((int **) train->data)[i][0]);
            inputs[batch][i] = curr;
            labels[batch][i] = create_vector(10);
            labels[batch][i]->data[label] = 1.f;

        }

    }
    */
    
    printf("Training with %d epochs, %d batch size, and %d batches across %d examples\n", epochs, batch_size, batches, total_data_len);


    /*
    for (size_t batch = 0; batch < batches; batch++) {

        for (size_t i = 0; i < total_data_len, i < (batch_size * (i + 1)); i ++) {

            destroy_vector(inputs[batch][i]);
            destroy_vector(labels[batch][i]);

        }
    }
    */
    destroy_csv_output(train);
    return 0;

}