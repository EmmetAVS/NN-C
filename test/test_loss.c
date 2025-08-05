#include "loss.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include "tests.h"

int main() {
    
    nnlib_startup();
    
    const BASE_TYPE expected[BATCH_SIZE][LENGTH] = {
        {1.0f, 0.0f, 0.5f, 0.8f, 0.2f},
        {0.4f, 0.9f, 0.1f, 0.6f, 0.0f},
        {0.3f, 0.7f, 0.2f, 1.0f, 0.5f}
    };

    const BASE_TYPE outputs[BATCH_SIZE][LENGTH] = {
        {0.9f, 0.1f, 0.4f, 0.7f, 0.3f},
        {0.5f, 1.0f, 0.2f, 0.4f, 0.1f},
        {0.2f, 0.6f, 0.3f, 0.9f, 0.6f}
    };

    Vector *expected_vecs[BATCH_SIZE];
    Vector *output_vecs[BATCH_SIZE];

    for (int i = 0; i < BATCH_SIZE; i++) {
        expected_vecs[i] = create_vector(LENGTH);
        output_vecs[i] = create_vector(LENGTH);
        for (int j = 0; j < LENGTH; j++) {
            expected_vecs[i]->data[j] = expected[i][j];
            output_vecs[i]->data[j] = outputs[i][j];
        }
    }

    //Test MSE loss

    Vector *mse_expected = expected_vecs[0];
    Vector *mse_output = output_vecs[0];

    BASE_TYPE mse_result = mean_squared_error_loss.forward(mse_output, mse_expected);
    assert(fabsf(mse_result - 0.01f) < TOLERANCE);

    assert(fabsf(calculate_cost(output_vecs, expected_vecs, BATCH_SIZE, mean_squared_error_loss.forward) - 0.012f) < TOLERANCE);

    Vector *dLoss_dActivated = mean_squared_error_loss.backward(mse_output, mse_expected);
    const BASE_TYPE expected_mse_grads_data[LENGTH] = {-0.04f, 0.04f, -0.04f, -0.04f, 0.04f};

    for (size_t i = 0; i < dLoss_dActivated->length; i ++) {

        assert(fabsf(expected_mse_grads_data[i] - dLoss_dActivated->data[i]) <= TOLERANCE);

    }

    for (size_t i = 0; i < BATCH_SIZE; i ++) {

        destroy_vector(expected_vecs[i]);
        destroy_vector(output_vecs[i]);

    }

    destroy_vector(dLoss_dActivated);

    return 0;

}