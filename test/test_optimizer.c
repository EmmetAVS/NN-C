#include "optimizer.h"
#include "layer.h"
#include "backprop.h"
#include "tests.h"
#include <assert.h>
#include <stdio.h>

void test_create_and_destroy_optimizer() {
    BASE_TYPE lr = 0.1f;

    Optimizer *opt = create_SGD_optimizer(lr);
    assert(opt != NULL);
    assert(opt->learning_rate == lr);

    Layer *l = create_layer(2, 3, activation_relu);
    LayerGradients *grads = create_layer_gradients(2, 3);

    for (size_t i = 0; i < 3; i++) {
        grads->d_biases->data[i] = 1.0f;
        for (size_t j = 0; j < 2; j++) {
            grads->d_weights->data[i * 2 + j] = 0.5f;
            l->weights->data[i * 2 + j] = 1.0f;
        }
        l->biases->data[i] = 1.0f;
    }

    opt->step(opt, l, grads);

    for (size_t i = 0; i < 3; i++) {
        assert(l->biases->data[i] == 1.0f - lr * 1.0f);
        for (size_t j = 0; j < 2; j++) {
            assert(l->weights->data[i * 2 + j] == 1.0f - lr * 0.5f);
        }
    }

    destroy_optimizer(opt);
    destroy_layer(l);
    destroy_layer_gradients(grads);
}

int main() {
    
    nnlib_startup();
    
    test_create_and_destroy_optimizer();
    return 0;
}
