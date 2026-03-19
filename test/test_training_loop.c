#include "tests.h"
#include "model2d.h"
#include "optimizer.h"
#include "loss.h"
#include "layer.h"

#define SAMPLES 4
#define EPOCHS 200
#define PRINT_EVERY 50

Vector *create_vector_from_array(BASE_TYPE *data, size_t size) {
    Vector *v = create_vector(size);
    for (size_t i = 0; i < size; ++i)
        v->data[i] = data[i];
    return v;
}

void print_loss(BASE_TYPE loss, int step) {
    printf("Step %d: Loss = %.6f\n", step, loss);
}

void test_training_loop_reduces_loss() {
    BASE_TYPE x_data[4][2] = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    BASE_TYPE y_data[4][1] = {
        {0}, {1}, {1}, {0}
    };

    size_t *shape[2];
    shape[0] = malloc(2 * sizeof(size_t));
    shape[1] = malloc(2 * sizeof(size_t));
    shape[0][0] = 2; shape[0][1] = 4;
    shape[1][0] = 4; shape[1][1] = 1;

    ActivationFunction activations[2] = {activation_relu, activation_sigmoid};
    Model2D *model = create_model2d(shape, activations, 2, mean_squared_error_loss);
    model2d_set_calculate_grads(model, true);

    Optimizer2D *opt = create_SGD_optimizer2d(1.f);

    Vector *inputs[SAMPLES], *labels[SAMPLES];
    for (int i = 0; i < SAMPLES; ++i) {
        inputs[i] = create_vector_from_array(x_data[i], 2);
        labels[i] = create_vector_from_array(y_data[i], 1);
    }

    BASE_TYPE prev_loss = 0;

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        model2d_zero_grads(model);
        model2d_set_max_grads(model, SAMPLES);

        for (int i = 0; i < SAMPLES; ++i) {
            Vector *output = model2d_forward(model, inputs[i]);
            model2d_backward(model, labels[i]);
            destroy_vector(output);
        }

        model2d_average_grads(model);
        model2d_step(model, opt);

        if (epoch % PRINT_EVERY == 0 || epoch == EPOCHS - 1) {
            BASE_TYPE total_loss = 0;
            for (int i = 0; i < SAMPLES; ++i) {
                model2d_set_calculate_grads(model, false);
                Vector *output = model2d_forward(model, inputs[i]);
                model2d_set_calculate_grads(model, true);
                total_loss += mean_squared_error_loss.forward(output, labels[i]);
                destroy_vector(output);
            }
            total_loss /= SAMPLES;

            print_loss(total_loss, epoch);

            if (epoch == 0) {
                prev_loss = total_loss;
            } else {
                assert(total_loss < prev_loss && "Loss did not decrease.");
                prev_loss = total_loss;
            }
        }
    }

    for (int i = 0; i < SAMPLES; ++i) {
        destroy_vector(inputs[i]);
        destroy_vector(labels[i]);
    }
    destroy_optimizer2d(opt);
    destroy_model2d(model);
    free(shape[0]);
    free(shape[1]);

    printf("Training loop test passed: Loss decreased over time.\n");
}

int main() {
    
    nnlib_startup();
    
    test_training_loop_reduces_loss();
    return 0;
}
