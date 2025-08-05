#include "tests.h"
#include "model.h"
#include "optimizer.h"
#include "loss.h"
#include "layer.h"

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

    const int samples = 4;
    const int epochs = 200;
    const int print_every = 50;

    size_t *shape[2];
    shape[0] = malloc(2 * sizeof(size_t));
    shape[1] = malloc(2 * sizeof(size_t));
    shape[0][0] = 2; shape[0][1] = 4;
    shape[1][0] = 4; shape[1][1] = 1;

    ActivationFunction activations[2] = {activation_relu, activation_sigmoid};
    Model *model = create_model(shape, activations, 2, mean_squared_error_loss);
    model_set_calculate_grads(model, true);

    Optimizer *opt = create_SGD_optimizer(1.f);

    Vector *inputs[samples], *labels[samples];
    for (int i = 0; i < samples; ++i) {
        inputs[i] = create_vector_from_array(x_data[i], 2);
        labels[i] = create_vector_from_array(y_data[i], 1);
    }

    BASE_TYPE prev_loss = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model_zero_grads(model);
        model_set_max_grads(model, samples);

        for (int i = 0; i < samples; ++i) {
            Vector *output = model_forward(model, inputs[i]);
            model_backward(model, labels[i]);
            destroy_vector(output);
        }

        model_average_grads(model);
        model_step(model, opt);

        if (epoch % print_every == 0 || epoch == epochs - 1) {
            BASE_TYPE total_loss = 0;
            for (int i = 0; i < samples; ++i) {
                model_set_calculate_grads(model, false);
                Vector *output = model_forward(model, inputs[i]);
                model_set_calculate_grads(model, true);
                total_loss += mean_squared_error_loss.forward(output, labels[i]);
                destroy_vector(output);
            }
            total_loss /= samples;

            print_loss(total_loss, epoch);

            if (epoch == 0) {
                prev_loss = total_loss;
            } else {
                assert(total_loss < prev_loss && "Loss did not decrease.");
                prev_loss = total_loss;
            }
        }
    }

    for (int i = 0; i < samples; ++i) {
        destroy_vector(inputs[i]);
        destroy_vector(labels[i]);
    }
    destroy_optimizer(opt);
    destroy_model(model);
    free(shape[0]);
    free(shape[1]);

    printf("Training loop test passed: Loss decreased over time.\n");
}

int main() {
    
    nnlib_startup();
    
    test_training_loop_reduces_loss();
    return 0;
}
