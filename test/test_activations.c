#include "activations.h"
#include "tests.h"

int test_softmax_cross_entropy() {
    Vector *logits = create_vector(3);
    Vector *labels = create_vector(3);
    Vector *output;

    logits->data[0] = 1.0f;
    logits->data[1] = 2.0f;
    logits->data[2] = 3.0f;

    labels->data[0] = 0.0f;
    labels->data[1] = 0.0f;
    labels->data[2] = 1.0f;

    BASE_TYPE loss = activation_loss_softmax_cross_entropy
        .function.activation_loss_function.forward_with_loss(logits, labels);

    assert(!isnan(loss));
    assert(fabsf(0.407606f - loss) < TOLERANCE);

    output = activation_loss_softmax_cross_entropy.function.activation_loss_function.forward(logits);

    BASE_TYPE sum = 0.0f;
    for (size_t i = 0; i < output->length; ++i) {
        sum += output->data[i];
    }
    assert(fabsf(sum - 1.0f) < TOLERANCE);

    Vector *grads = activation_loss_softmax_cross_entropy
        .function.activation_loss_function.backward(logits, labels);

    assert(grads != NULL);

    for (size_t i = 0; i < grads->length; ++i) {
        assert(!isnan(grads->data[i]));
    }

    destroy_vector(logits);
    destroy_vector(labels);
    destroy_vector(output);
    destroy_vector(grads);

    return 0;
}

int main() {

    //Test RELU
    BASE_TYPE ex_relu_logits[LENGTH] = {-0.1f, 0.2f, 0.4f, 0.1f};
    BASE_TYPE ex_relu_activated[LENGTH] = {0.f, 0.2f, 0.4f, 0.1f};
    BASE_TYPE ex_relu_backwards[LENGTH] = {0.f, 1.0f, 1.0f, 1.0f};

    Vector *logits = create_vector(LENGTH);

    free(logits->data);
    logits->data = ex_relu_logits;

    Vector *output = activation_relu.function.activation_function.forward(logits);

    for (size_t i = 0; i < LENGTH; i ++) {

        assert(fabsf(output->data[i] - ex_relu_activated[i]) < TOLERANCE);

    }

    Vector *relu_backwards = activation_relu.function.activation_function.backward(logits, output);

    for (size_t i = 0; i < LENGTH; i ++) {

        assert(fabsf(relu_backwards->data[i] - ex_relu_backwards[i]) < TOLERANCE);

    }

    //Test Softmax + Cross Entropy Combination
    test_softmax_cross_entropy();
    free(logits);
    destroy_vector(output);
    destroy_vector(relu_backwards);

    return 0;

}