#include "model.h"
#include "optimizer.h"
#include "data.h"
#include <stdio.h>

int main() {

    const int epochs = 2;
    const int batch_size = 128;

    CSVOutput *train = read_csv("data/mnist_train.csv", INTEGER);
    if (!train) {
        printf("Train dataset failed to load.\n");
        return 0;
    }
    
    int total_data_len = (int) train->data_rows;
    const int batches = total_data_len / batch_size;

    printf("Training with %d epochs, %d batch size, and %d batches across %d examples\n", epochs, batch_size, batches, total_data_len);

    destroy_csv_output(train);
    return 0;

}