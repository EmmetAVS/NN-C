#ifndef DATA_H
#define DATA_H

#include <stddef.h>

typedef enum CSVOutputType {

    INTEGER, STRING

} CSVOutputType;

typedef struct CSVOutput {
    CSVOutputType output_type;
    char **col_names;
    void **data;
    size_t data_rows;
    size_t cols;
} CSVOutput;

CSVOutput *read_csv(const char *filename, CSVOutputType output_type);
void destroy_csv_output(CSVOutput *output);

#endif