#include "data.h"
#include "tests.h"

int main() {

    CSVOutput *output = read_csv("/mnt/c/Users/aarus/Downloads/test_data.csv", FLOAT);
    printf("output: %p\n", output);

    printf("Rows: %zu, Cols: %zu.\n", output->data_rows, output->cols);
    printf("Col names: \n");
    for (size_t i = 0; i < output->cols; i ++) {

        printf("%s, ", output->col_names[i]);

    }

    printf("\n\nData: \n");

    for (size_t r = 0; r < output->data_rows; r ++) {

        for (size_t c = 0; c < output->cols; c ++) {

            printf("%f", ((float**) (output->data))[r][c]);
            if (c != output->cols - 1) printf(", ");

        }

        printf("\n");

    }

    destroy_csv_output(output);

    return 0;

}