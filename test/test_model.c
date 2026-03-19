#include "tests.h"
#include "model2d.h"
#include "activations.h"
#include "loss.h"
#include "layer.h"
#include "types.h"

void test_model_forward_backward() {
    size_t *shapes[] = {
        (size_t[]) {2, 2},
        (size_t[]) {2, 1}
    };

    ActivationFunction activations[] = {
        activation_relu,
        activation_loss_softmax_cross_entropy
    };

    Model2D *model = create_model2d(shapes, activations, 2, mean_squared_error_loss);

    model2d_set_calculate_grads(model, true);
    model2d_set_max_grads(model, 10);

    Vector *input = create_vector(2);
    input->data[0] = 1.0;
    input->data[1] = -1.0;

    Vector *label = create_vector(1);
    label->data[0] = 1.0;

    Vector *output = model2d_forward(model, input);
    assert(output != NULL);

    model2d_backward(model, label);
    assert(model->current_grads_accumulated == 1);

    model2d_average_grads(model);

    for (size_t i = 0; i < model->num_layers; ++i) {
        assert(model->averaged_gradients[i] != NULL);
        assert(model->averaged_gradients[i]->d_weights != NULL);
        assert(model->averaged_gradients[i]->d_biases != NULL);
    }

    destroy_vector(input);
    destroy_vector(label);
    destroy_vector(output);
    destroy_model2d(model);
}

int main() {
    
    nnlib_startup();
    
    test_model_forward_backward();
    return 0;
}
