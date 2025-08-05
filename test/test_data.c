#include "data.h"
#include "tests.h"

int main() {

    CSVOutput *output = read_csv("../../examples/digit_recognition/data/mnist.csv", FLOAT);
    printf("output: %p\n", output);

    printf("Rows: %zu, Cols: %zu.\n", output->data_rows, output->cols);

    printf("\n\nData: \n");

    const size_t rows = output->data_rows > 5 ? 5 : output->data_rows;
    const size_t cols = output->cols > 5 ? 5 : output->cols;
    for (size_t c = 0; c < cols; c ++) {

            printf("%s", (output->col_names)[c]);
            if (c != cols - 1) printf(", ");

        }

    if (cols != output->cols) printf("...");
    printf("\n");
    
    for (size_t r = 0; r < rows; r ++) {

        for (size_t c = 0; c < cols; c ++) {

            printf("%f", ((float**) (output->data))[r][c]);
            if (c != cols - 1) printf(", ");

        }

        if (cols != output->cols) printf("...");
        printf("\n");

    }
    if (rows != output->data_rows) printf("...\n");

    destroy_csv_output(output);

    return 0;

}