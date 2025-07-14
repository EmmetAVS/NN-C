#include "loss.h"
#include "types.h"
#include <stdio.h>
#include <assert.h>

#define BATCH_SIZE 3
#define LENGTH 5
#define TOLERANCE 0.0001f

int main() {
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
    assert(abs(mse_result - 0.01f) < TOLERANCE);

    assert(abs(calculate_cost(output_vecs, expected_vecs, BATCH_SIZE, mean_squared_error_loss.forward) - 0.012f) < TOLERANCE);

    return 0;

}