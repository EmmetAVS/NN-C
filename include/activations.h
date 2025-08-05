#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "types.h"
#include "loss.h"

typedef Vector* (*ActivationFunctionForward)(Vector *logits);
typedef Vector* (*ActivationFunctionBackward)(Vector *logits, Vector *output);

typedef struct RawActivationFunction {

    ActivationFunctionForward forward;
    ActivationFunctionBackward backward;

} RawActivationFunction;

typedef BASE_TYPE (*ActivationLossFunctionForward)(Vector *logits, Vector *labels);
typedef Vector* (*ActivationLossFunctionBackward)(Vector *logits, Vector *labels);

typedef struct ActivationLossFunction {
    
    ActivationFunctionForward forward;
    ActivationLossFunctionForward forward_with_loss;
    ActivationLossFunctionBackward backward;

} ActivationLossFunction;

typedef enum ActivationFunctionType {
    RAW, WITH_LOSS
} ActivationFunctionType;

typedef union BaseActivationFunction {
    
    RawActivationFunction activation_function;
    ActivationLossFunction activation_loss_function;

} BaseActivationFunction;

typedef struct ActivationFunction {

    ActivationFunctionType type;
    BaseActivationFunction function;

} ActivationFunction;

extern ActivationFunction activation_relu;
extern ActivationFunction activation_sigmoid;
extern ActivationFunction activation_loss_softmax_cross_entropy;

void _init_activations();

#endif