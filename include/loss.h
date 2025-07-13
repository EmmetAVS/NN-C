#ifndef LOSS_H
#define LOSS_H

#include "types.h"

typedef BASE_TYPE (*LossFunction)(Vector *, Vector *);

BASE_TYPE mean_squared_error_loss(Vector *outputs, Vector *expected);
BASE_TYPE cross_entropy_loss(Vector *outputs, Vector *expected);
BASE_TYPE calculate_cost(Vector **outputs, Vector **expected, size_t batch_size, LossFunction loss_function);

#endif