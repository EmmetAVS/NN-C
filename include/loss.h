#ifndef LOSS_H
#define LOSS_H

#include "types.h"

typedef BASE_TYPE (*LossFunctionForward)(Vector *inputs, Vector *labels);
typedef Vector* (*LossFunctionBackward)(Vector *inputs, Vector *labels);

typedef struct LossFunction {

    LossFunctionForward forward;
    LossFunctionBackward backward;

} LossFunction;

extern LossFunction mean_squared_error_loss;
extern LossFunctionForward cross_entropy_loss;

BASE_TYPE calculate_cost(Vector **inputs_list, Vector **labels_list, size_t batch_size, LossFunctionForward loss_function);

#endif