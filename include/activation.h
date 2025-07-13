#ifndef ACTIVATION_H
#define ACTIVATION_H

#include "types.h"
Vector* relu(Vector* input);
Vector* sigmoid(Vector* input);
Vector* softmax(Vector* input);
Vector* flatten(Matrix* input);

#endif