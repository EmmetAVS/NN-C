#include "activations.h"
#include "tests.h"

int main() {

    //Test RELU
    BASE_TYPE ex_relu_logits[LENGTH] = {-0.1f, 0.2f, 0.4f, 0.1f};
    BASE_TYPE ex_relu_activated[LENGTH] = {0.f, 0.2f, 0.4f, 0.1f};
    BASE_TYPE ex_relu_backwards[LENGTH] = {0.f, 1.0f, 1.0f, 1.0f};

    Vector *logits = create_vector(LENGTH);

    logits->data = ex_relu_logits;

    Vector *output = activation_relu.function.activation_function.forward(logits);

    for (size_t i = 0; i < LENGTH; i ++) {

        assert(abs(output->data[i] - ex_relu_activated[i]) < TOLERANCE);

    }

    Vector *relu_backwards = activation_relu.function.activation_function.backward(logits, output);

    for (size_t i = 0; i < LENGTH; i ++) {

        assert(abs(relu_backwards->data[i] - ex_relu_backwards[i]) < TOLERANCE);

    }

    //Test Softmax + Cross Entropy Combination
    

    return 0;

}