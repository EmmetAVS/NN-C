#include "tests.h"
#include "model.h"
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

Model *train_model(size_t **shape, ActivationFunction *activations) {

    Model *model = create_model(shape, activations, 2, mean_squared_error_loss);
    model_set_calculate_grads(model, true);

    Optimizer *opt = create_SGD_optimizer(1.f);

    inputs = (Vector **)malloc(sizeof(Vector *) * samples);
    labels = (Vector **)malloc(sizeof(Vector *) * samples);

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
    }

    destroy_optimizer(opt);
    return model;
}

int main() {

    const char *filename = "test_serialization_model_params";

    size_t *shape[2];
    shape[0] = malloc(2 * sizeof(size_t));
    shape[1] = malloc(2 * sizeof(size_t));
    shape[0][0] = 2; shape[0][1] = 4;
    shape[1][0] = 4; shape[1][1] = 1;

    ActivationFunction activations[2] = {activation_relu, activation_sigmoid};
    
    Model *model1 = train_model(shape, activations);
    write_model_params(model1, filename);
    Model *model2 = create_model(shape, activations, model1->num_layers, model1->loss);
    load_model_params(model2, filename);

    model_set_calculate_grads(model1, false);
    model_set_calculate_grads(model2, false);

    BASE_TYPE total_loss_difference = 0;

    for (int i = 0; i < samples; ++i) {
        Vector *output = model_forward(model1, inputs[i]);
        BASE_TYPE Model1Loss = mean_squared_error_loss.forward(output, labels[i]);

        destroy_vector(output);

        output = model_forward(model2, inputs[i]);
        BASE_TYPE Model2Loss = mean_squared_error_loss.forward(output, labels[i]);

        total_loss_difference += fabs(Model1Loss - Model2Loss);

        destroy_vector(output);
        printf("Model 1 Loss: %f, Model 2 Loss: %f\n", Model1Loss, Model2Loss);

    }

    printf("Total loss difference: %f\n", total_loss_difference);
    assert(total_loss_difference < EPSILON);

    destroy_model(model1);
    destroy_model(model2);
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
