#include "data.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int count_lines(FILE *file) {
    if (file == NULL) return -1;

    int lines = 0;
    int cur_char;

    while ((cur_char = fgetc(file)) != EOF) {
        if (cur_char == '\n') {
            lines++;
        }
    }

    fseek(file, -1, SEEK_END);
    if (fgetc(file) != '\n') {
        lines++;
    }

    return lines;
}

static char *read_full_line(FILE *file, char *buffer, size_t *buffer_size) {
    if (file == NULL) {
        free(buffer);
        return NULL;
    }

    long start_pos = ftell(file);
    if (start_pos == -1L) {
        free(buffer);
        return NULL;
    }

    int cur_char;
    size_t length = 0;

    while ((cur_char = fgetc(file)) != '\n' && cur_char != EOF) {
        length++;
    }

    if (cur_char == '\n') {
        length++;
    }

    if (length == 0 && cur_char == EOF) {
        free(buffer);
        return NULL;
    }

    fseek(file, start_pos, SEEK_SET);

    if (*buffer_size < length + 1) {
        free(buffer);
        buffer = (char *) malloc(sizeof(char) * (length + 1));
        *buffer_size = length + 1;
    }
    if (!buffer) 
        return NULL;

    if (fread(buffer, sizeof(char), length, file) != length) {
        free(buffer);
        return NULL;
    }

    buffer[length] = '\0';

    if (length > 0 && buffer[length - 1] == '\n') {
        buffer[length - 1] = '\0';
    } 
    
    if (length > 1 && buffer[length - 2] == '\r') {
        buffer[length - 2] = '\0';
    }

    return buffer;
}

static char **parse_strings_from_line(char *line, size_t num_cols) {

    char **data = (char **)malloc(sizeof(char*) * num_cols);

    int i = 0;
    int cur_col = 0;
    while (cur_col < num_cols) {
        
        int j = i;
        while (line[j] != ',' && line[j] != '\0' && line[j] != '\n') {
            j++;
        }

        int cur_item_length = j - i;
        char *cur_item = (char *)malloc(sizeof(char) * (cur_item_length + 1));
        cur_item[cur_item_length] = '\0';
        for (j = 0; j < cur_item_length; j ++) {
            cur_item[j] = line[i + j];
        }

        data[cur_col] = cur_item;
        cur_col++;
        i = cur_item_length + i + 1;

    }

    return data;

}

static int *parse_ints_from_line(char *line, size_t num_cols) {

    int *data = (int *)malloc(sizeof(int) * num_cols);
    
    char **strings = parse_strings_from_line(line, num_cols);
    for (size_t i = 0; i < num_cols; i ++) {

        data[i] = atoi(strings[i]);
        free(strings[i]);

    }

    free(strings);
    return data;

}

static float *parse_floats_from_line(char *line, size_t num_cols) {

    float *data = (float *)malloc(sizeof(float) * num_cols);
    
    char **strings = parse_strings_from_line(line, num_cols);
    for (size_t i = 0; i < num_cols; i ++) {

        data[i] = (float) atof(strings[i]);
        free(strings[i]);

    }

    free(strings);
    return data;

}

CSVOutput *read_csv(const char *filename, CSVOutputType output_type) {

    FILE *file = fopen(filename, "r");

    if (file == NULL) {
        return NULL;
    }

    int lines = count_lines(file);
    fseek(file, 0, SEEK_SET);

    size_t current_buffer_size = 1;
    char *buffer = (char *)malloc(sizeof(char) * current_buffer_size);
    buffer = read_full_line(file, buffer, &current_buffer_size);

    if (buffer == NULL) {
        return NULL;
    }

    CSVOutput *output = (CSVOutput *)malloc(sizeof(CSVOutput));
    output->output_type = output_type;

    output->data_rows = lines - 1;
    output->data = (void **)malloc(sizeof(void *) * output->data_rows);

    int i = 0;
    int cols = 1;
    while (buffer[i] != '\0') {
        if (buffer[i] == ',') cols++;
        i++;
    }

    output->cols = cols;
    output->col_names = parse_strings_from_line(buffer, cols);
    buffer = read_full_line(file, buffer, &current_buffer_size);
    
    size_t cur_row = 0;
    while (buffer != NULL) {

        if (output_type == INTEGER) {
            output->data[cur_row] = parse_ints_from_line(buffer, output->cols);
        } else if (output_type == STRING) {
            output->data[cur_row] = parse_strings_from_line(buffer, output->cols);
        } else if (output_type == FLOAT) {
            output->data[cur_row] = parse_floats_from_line(buffer, output->cols);
        }
        cur_row++;

        buffer = read_full_line(file, buffer, &current_buffer_size);

    }

    free(buffer);
    fclose(file);
    return output;

}

void destroy_csv_output(CSVOutput *output) {

    for (size_t rows = 0; rows < output->data_rows; rows ++) {

        if (output->output_type != STRING) {
            free(output->data[rows]);
            continue;
        }

        for (size_t cols = 0; cols < output->cols; cols ++) {
            free(((void ***) output->data)[rows][cols]);
        }
        
        free(output->data[rows]);

    }
    
    free(output->data);
    for (size_t cols = 0; cols < output->cols; cols ++) {

        free(output->col_names[cols]);

    }

    free(output->col_names);
    free(output);

}