#include "backprop.h"
#include "types.h"
#include "operations.h"
#include <stdlib.h>
#include <assert.h>

Vector *forward_layer(Layer *layer, Vector *input, bool save_context) {

    Vector *logits_no_bias = multiply_matrix_with_vector(layer->weights, input);
    Vector *logits = add_vector_to_vector(logits_no_bias, layer->biases);
    destroy_vector(logits_no_bias);

    const ActivationFunctionForward forward_activation = layer->activation.type == RAW ? layer->activation.function.activation_function.forward : layer->activation.function.activation_loss_function.forward;
    Vector *activated = forward_activation(logits);

    if (layer->context) destroy_layer_context(layer->context);
    
    if (save_context) {
        layer->context = create_layer_context(input, logits, activated);
    } else {
        layer->context = NULL;
        destroy_vector(logits);
    }

    return activated;

}

LayerGradients *backward_layer(Layer *layer, Vector *input, Vector *logits, BackpropContext *context) {

    LayerGradients *grads = create_layer_gradients(layer->input_size, layer->output_size);

    //Calculate dLoss/dLogits
    Vector *dLoss_dLogits;
    if (context->type == dLoss_dActivation) {

        BackpropContextDLossDActivation dLoss_dActivation_context = context->dLoss_dActivation;
        Vector *dLoss_dActivation = dLoss_dActivation_context.dL_dA;
        Vector *output = dLoss_dActivation_context.output;

        Vector *dActivation_dLogits = layer->activation.function.activation_function.backward(logits, output);

        dLoss_dLogits = multiply_vector_contents(dLoss_dActivation, dActivation_dLogits);
        
    } else if (context->type == LabelsOutput) {

        BackpropContextLabelsOutput labels_outputs = context->labels_output;
        ActivationLossFunctionBackward activation_loss_backward = layer->activation.function.activation_loss_function.backward;
        dLoss_dLogits = activation_loss_backward(logits, labels_outputs.labels);

    } else return NULL;

    //Calculate weight gradient
    for (size_t output_index = 0; output_index < logits->length; output_index++) {

        for (size_t input_index = 0; input_index < input->length; input_index ++) {

            BASE_TYPE value = dLoss_dLogits->data[output_index] * input->data[input_index];
            matrix_set_value_at(grads->d_weights, output_index, input_index, value);

        }

    }

    //Calculate bias gradient
    for (size_t output_index = 0; output_index < logits->length; output_index ++) {

        BASE_TYPE value = dLoss_dLogits->data[output_index];
        grads->d_biases->data[output_index] = value;
        
    }

    //Calculate previous activation gradient (dC/dA(L-1))
    for (size_t input_index = 0; input_index < input->length; input_index ++) {

        BASE_TYPE sum = 0;
        for (size_t output_index = 0; output_index < logits->length; output_index ++) {
            BASE_TYPE weight_value = matrix_get_value_at(layer->weights, output_index, input_index);
            sum += weight_value * dLoss_dLogits->data[output_index];
        }
        grads->d_inputs->data[input_index] = sum;

    }

    destroy_vector(dLoss_dLogits);

    return grads;

}

void update_layer_parameters(Layer *layer, LayerGradients *grads, BASE_TYPE learning_rate) {

    //Update weights
    for (size_t output_index = 0; output_index < layer->output_size; output_index ++) {

        for (size_t input_index = 0; input_index < layer->input_size; input_index ++) {

            const size_t index = (output_index * layer->input_size) + input_index;
            layer->weights->data[index] -= learning_rate * grads->d_weights->data[index];

        }

    }

    //Update biases
    for (size_t output_index = 0; output_index < layer->output_size; output_index ++) {

        layer->biases->data[output_index] -= grads->d_biases->data[output_index] * learning_rate;

    }

}

LayerGradients *create_layer_gradients(size_t input_size, size_t output_size) {

    LayerGradients *grads = (LayerGradients*)malloc(sizeof(LayerGradients));

    grads->d_biases = create_vector(output_size);
    grads->d_inputs = create_vector(input_size);
    grads->d_weights = create_matrix(output_size, input_size);

    return grads;

}

void destroy_layer_gradients(LayerGradients *grads) {

    destroy_vector(grads->d_biases);
    destroy_vector(grads->d_inputs);
    destroy_matrix(grads->d_weights);

    free(grads);

}

LayerGradients *average_gradients(LayerGradients **grads, size_t batch_size) {

    //Get lists of matrices/vectors
    Vector **d_biases = (Vector **)malloc(sizeof(Vector *) * batch_size);
    Vector **d_inputs = (Vector **)malloc(sizeof(Vector *) * batch_size);
    Matrix **d_weights = (Matrix **)malloc(sizeof(Matrix *) * batch_size);

    for (size_t i = 0; i < batch_size; i ++) {

        d_biases[i] = grads[i]->d_biases;
        d_inputs[i] = grads[i]->d_inputs;
        d_weights[i] = grads[i]->d_weights;

    }

    LayerGradients *averaged = (LayerGradients *)malloc(sizeof(LayerGradients));

    averaged->d_biases = average_vectors(d_biases, batch_size);
    averaged->d_inputs = average_vectors(d_inputs, batch_size);
    averaged->d_weights = average_matrices(d_weights, batch_size);
        
    return averaged;
}