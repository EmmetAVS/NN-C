#include "model.h"
#include <stdlib.h>

Model *create_model(size_t **shape, ActivationFunction *activations, size_t num_layers, LossFunction loss) {

    Model *model = (Model *)malloc(sizeof(Model));
    model->num_layers = num_layers;
    model->layers = (Layer **)malloc(sizeof(Layer *) * num_layers);
    for (size_t i = 0; i < num_layers; i ++) {

        model->layers[i] = create_layer(shape[i][0], shape[i][1], activations[i]);

    }

    model->loss = loss;
    model->calc_grads = true;
    model->current_grads_accumulated = 0;
    model->max_grads = 0;
    model->gradients = NULL;

    return model;

}

void destroy_model(Model *model) {

    model_zero_grads(model);

    for (size_t i = 0; i < model->num_layers; i ++) {

        destroy_layer(model->layers[i]);

    }

    free(model->layers);
    free(model);

}

void model_set_calculate_grads(Model *model, bool calc_grads) {

    model->calc_grads = calc_grads;

}

void model_zero_grads(Model *model) {

    if (!model->gradients) return;
    for (size_t i = 0; i < model->num_layers; i ++) {

        for (size_t j = 0; j < model->current_grads_accumulated; j++) {
            destroy_layer_gradients(model->gradients[i][j]);
        }
        if (model->averaged_gradients) destroy_layer_gradients(model->averaged_gradients[i]);

    }

    free(model->gradients);
    model->current_grads_accumulated = 0;
    model->max_grads = 0;

}

void model_set_max_grads(Model *model, size_t max_grads) {
    model_zero_grads(model);

    model->gradients = (LayerGradients ***)malloc(sizeof(LayerGradients**) * model->num_layers);
    model->max_grads = max_grads;

    for (size_t i = 0; i < model->num_layers; i ++) {

        model->gradients[i] = (LayerGradients **)malloc(sizeof(LayerGradients*) * max_grads);

    }
}

static Vector *model_inference(Model *model, Vector *inputs) {

    //Assume model no grad for inference

    Vector *current_output = inputs;

    for (size_t i = 0; i < model->num_layers; i++) {

        current_output = forward_layer(model->layers[i], current_output, false);

    }

    return current_output;

}

static Vector *model_forward_with_grad(Model *model, Vector *inputs) {

    Vector *current_output = inputs;

    for (size_t i = 0; i < model->num_layers; i++) {

        current_output = forward_layer(model->layers[i], current_output, true);

    }

    return current_output;

}

Vector *model_forward(Model *model, Vector *inputs) {

    if (model->calc_grads) {
        return model_inference(model, inputs);
    }

    return model_forward_with_grad(model, inputs);

}

void model_backward(Model *model, Vector *labels) {

    if (!model->calc_grads || model->num_layers < 1 || model->current_grads_accumulated == model->max_grads) return;
    Vector *dL_dA = NULL;

    for (size_t i = model->num_layers - 1; i >= 0; i --) {

        Layer *current_layer = model->layers[i];
        LayerContext *layer_context = current_layer->context;

        BackpropContext backprop_context;

        if (current_layer->activation.type == RAW) {

            if (dL_dA == NULL) {

                dL_dA = model->loss.backward(layer_context->activated_output, labels);

            }

            backprop_context.type = dLoss_dActivation;
            BackpropContextDLossDActivation dLoss_dActivation_context = {.output = layer_context->activated_output, .dL_dA = dL_dA};
            backprop_context.dLoss_dActivation = dLoss_dActivation_context;

        } else {

            backprop_context.type = LabelsOutput;
            BackpropContextLabelsOutput labels_output_context = {.labels = labels};
            backprop_context.labels_output = labels_output_context;

        }

        model->gradients[i][model->current_grads_accumulated] = backward_layer(current_layer, layer_context->inputs, layer_context->logits, &backprop_context);
        dL_dA = model->gradients[i][model->current_grads_accumulated]->d_inputs;

    }
    model->current_grads_accumulated ++;

}

void model_average_grads(Model *model) {
    size_t grad_accumulation_diff = model->max_grads - model->current_grads_accumulated;

    model->averaged_gradients = (LayerGradients **)malloc(sizeof(LayerGradients*) * model->num_layers);
    for (size_t i = 0; i < model->num_layers; i ++) {

        LayerGradients **resized_accumulated_grads;
        if (grad_accumulation_diff > 0) {
            resized_accumulated_grads = (LayerGradients **)malloc(sizeof(LayerGradients *) * model->current_grads_accumulated);

            for (size_t j = 0; j < model->current_grads_accumulated; j ++) {
                resized_accumulated_grads[j] = model->gradients[i][j];
            }
        } else {
            resized_accumulated_grads = model->gradients[i];
        }

        model->averaged_gradients[i] = average_gradients(resized_accumulated_grads, model->current_grads_accumulated);
        if (grad_accumulation_diff > 0) free(resized_accumulated_grads);

    }

}

void model_clear_accumulated_grads(Model *model) {

    if (!model->gradients) return;
    for (size_t i = 0; i < model->num_layers; i ++) {

        for (size_t j = 0; j < model->current_grads_accumulated; j ++) {

            destroy_layer_gradients(model->gradients[i][j]);

        }

        free(model->gradients[i]);

    }

    free(model->gradients);
    model->current_grads_accumulated = 0;
    model->max_grads = 0;
    model->gradients = NULL;

}