#include "tests.h"
#include "model2d.h"
#include "optimizer.h"
#include "loss.h"
#include "layer.h"
#include "serialization.h"

const int samples = 4;
const int epochs = 200;

BASE_TYPE x_data[4][2] = {
    {0, 0}, {0, 1}, {1, 0}, {1, 1}
};
BASE_TYPE y_data[4][1] = {
    {0}, {1}, {1}, {0}
};

Vector **inputs, **labels;

Vector *create_vector_from_array(BASE_TYPE *data, size_t size) {
    Vector *v = create_vector(size);
    for (size_t i = 0; i < size; ++i)
        v->data[i] = data[i];
    return v;
}

Model2D *train_model(size_t **shape, ActivationFunction *activations) {

    Model2D *model = create_model2d(shape, activations, 2, mean_squared_error_loss);
    model2d_set_calculate_grads(model, true);

    Optimizer2D *opt = create_SGD_optimizer2d(1.f);

    inputs = (Vector **)malloc(sizeof(Vector *) * samples);
    labels = (Vector **)malloc(sizeof(Vector *) * samples);

    for (int i = 0; i < samples; ++i) {
        inputs[i] = create_vector_from_array(x_data[i], 2);
        labels[i] = create_vector_from_array(y_data[i], 1);
    }

    BASE_TYPE prev_loss = 0;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        model2d_zero_grads(model);
        model2d_set_max_grads(model, samples);

        for (int i = 0; i < samples; ++i) {
            Vector *output = model2d_forward(model, inputs[i]);
            model2d_backward(model, labels[i]);
            destroy_vector(output);
        }

        model2d_average_grads(model);
        model2d_step(model, opt);
    }

    destroy_optimizer2d(opt);
    return model;
}

int main() {
    
    nnlib_startup();

    const char *filename = "test_serialization_model_params";

    size_t *shape[2];
    shape[0] = malloc(2 * sizeof(size_t));
    shape[1] = malloc(2 * sizeof(size_t));
    shape[0][0] = 2; shape[0][1] = 4;
    shape[1][0] = 4; shape[1][1] = 1;

    ActivationFunction activations[2] = {activation_relu, activation_sigmoid};
    
    Model2D *model1 = train_model(shape, activations);
    write_model2d_params(model1, filename);
    Model2D *model2 = create_model2d(shape, activations, model1->num_layers, model1->loss);
    load_model2d_params(model2, filename);

    model2d_set_calculate_grads(model1, false);
    model2d_set_calculate_grads(model2, false);

    BASE_TYPE total_loss_difference = 0;

    for (int i = 0; i < samples; ++i) {
        Vector *output = model2d_forward(model1, inputs[i]);
        BASE_TYPE Model1Loss = mean_squared_error_loss.forward(output, labels[i]);

        destroy_vector(output);

        output = model2d_forward(model2, inputs[i]);
        BASE_TYPE Model2Loss = mean_squared_error_loss.forward(output, labels[i]);

        total_loss_difference += fabs(Model1Loss - Model2Loss);

        destroy_vector(output);
        printf("Model 1 Loss: %f, Model 2 Loss: %f\n", Model1Loss, Model2Loss);

    }

    printf("Total loss difference: %f\n", total_loss_difference);
    assert(total_loss_difference < EPSILON);

    destroy_model2d(model1);
    destroy_model2d(model2);
    free(shape[0]);
    free(shape[1]);

    for (int i = 0; i < samples; ++i) {
        destroy_vector(inputs[i]);
        destroy_vector(labels[i]);
    }

    free(inputs);
    free(labels);
    return 0;
}
