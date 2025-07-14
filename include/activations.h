#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "types.h"
#include "loss.h"

typedef Vector* (*ActivationFunctionForward)(Vector *logits);
typedef Vector* (*ActivationFunctionBackward)(Vector *logits, Vector *output);

typedef struct ActivationFunction {

    ActivationFunctionForward forward;
    ActivationFunctionBackward backward;

} ActivationFunction;

typedef BASE_TYPE (*ActivationLossFunctionForward)(Vector *logits, Vector *labels);
typedef Vector* (*ActivationLossFunctionBackward)(Vector *logits, Vector *labels);

typedef struct ActivationLossFunction {
    
    ActivationFunctionForward forward;
    ActivationLossFunctionForward forward_with_loss;
    ActivationLossFunctionBackward backward;

} ActivationLossFunction;

extern ActivationFunction activation_relu;
extern ActivationFunction activation_sigmoid;
extern ActivationLossFunction activation_loss_softmax_cross_entropy;

Vector* flatten(Matrix* input);

#endif