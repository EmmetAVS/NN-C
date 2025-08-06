#include "optimizer.h"
#include "model.h"
#include <stdlib.h>

static void optimizer_step_adam(Optimizer *o, Layer *layer, LayerGradients *grads) {

    OptimizerAdamState *state = (OptimizerAdamState *)o->optimizer_state;
    OptimizerAdamParams *params = (OptimizerAdamParams *)o->additional_parameters;

    int layer_index = -1;

    for (size_t i = 0; i < state->layers; i ++) {

        if (state->model->layers[i] == layer) {
            layer_index = (int) i;
            break;
        }

    }

    if (layer_index == -1) {
        return;
    }

    state->steps[layer_index] += 1;

    const BASE_TYPE beta_1_pow = pow(params->beta_1, state->steps[layer_index]);
    const BASE_TYPE beta_2_pow = pow(params->beta_2, state->steps[layer_index]);

    //Update weights
    for (size_t output_index = 0; output_index < layer->output_size; output_index ++) {

        for (size_t input_index = 0; input_index < layer->input_size; input_index ++) {

            const size_t index = (output_index * layer->input_size) + input_index;

            BASE_TYPE grad = grads->d_weights->data[index];

            BASE_TYPE m_t = (params->beta_1 * state->weight_m_t[layer_index]->data[index]) + ((1 - params->beta_1) * grad);
            BASE_TYPE v_t = (params->beta_2 * state->weight_v_t[layer_index]->data[index]) + ((1 - params->beta_2) * (grad * grad));

            state->weight_m_t[layer_index]->data[index] = m_t;
            state->weight_v_t[layer_index]->data[index] = v_t;

            //Correct bias
            BASE_TYPE no_bias_m_t = (m_t) / (1 - beta_1_pow);
            BASE_TYPE no_bias_v_t = (v_t) / (1 - beta_2_pow);

            layer->weights->data[index] -= o->learning_rate * (no_bias_m_t / (sqrt(no_bias_v_t) + EPSILON));

        }

    }

    //Update biases
    for (size_t output_index = 0; output_index < layer->output_size; output_index ++) {

        BASE_TYPE grad = grads->d_biases->data[output_index];

        BASE_TYPE m_t = (params->beta_1 * state->bias_m_t[layer_index]->data[output_index]) + ((1 - params->beta_1) * grad);
        BASE_TYPE v_t = (params->beta_2 * state->bias_v_t[layer_index]->data[output_index]) + ((1 - params->beta_2) * (grad * grad));

        state->bias_m_t[layer_index]->data[output_index] = m_t;
        state->bias_v_t[layer_index]->data[output_index] = v_t;

        //Correct bias
        BASE_TYPE no_bias_m_t = (m_t) / (1 - beta_1_pow);
        BASE_TYPE no_bias_v_t = (v_t) / (1 - beta_2_pow);

        layer->biases->data[output_index] -= o->learning_rate * (no_bias_m_t / (sqrt(no_bias_v_t) + EPSILON));

    }

}

static void optimizer_step_sgd(Optimizer *o, Layer *l, LayerGradients *grads) {

    update_layer_parameters(l, grads, o->learning_rate);

}

Optimizer *create_SGD_optimizer(BASE_TYPE learning_rate) {
    
    Optimizer* o = (Optimizer *)malloc(sizeof(Optimizer));
    o->learning_rate = learning_rate;
    o->step = optimizer_step_sgd;
    o->additional_parameters = NULL;
    o->optimizer_state = NULL;
    o->destruction = NULL;

    return o;
}

static void destroy_ADAM_optimizer(Optimizer *o) {

    OptimizerAdamState *state = (OptimizerAdamState *) (o->optimizer_state);

    for (size_t i = 0; i < state->layers; i ++) {

        destroy_matrix(state->weight_m_t[i]);
        destroy_matrix(state->weight_v_t[i]);

        destroy_vector(state->bias_m_t[i]);
        destroy_vector(state->bias_v_t[i]);

    }

    free(state->weight_m_t);
    free(state->weight_v_t);
    free(state->bias_m_t);
    free(state->bias_v_t);
    free(state->steps);

    free((OptimizerAdamParams *) (o->additional_parameters));
    free((OptimizerAdamState *) (o->optimizer_state));

}

Optimizer *create_ADAM_optimizer(Model *model, BASE_TYPE learning_rate, BASE_TYPE beta_1, BASE_TYPE beta_2) {

    Optimizer* o = (Optimizer *)malloc(sizeof(Optimizer));
    o->learning_rate = learning_rate;
    o->step = optimizer_step_adam;

    OptimizerAdamParams *params = (OptimizerAdamParams *)malloc(sizeof(OptimizerAdamParams));
    OptimizerAdamState *state = (OptimizerAdamState *)malloc(sizeof(OptimizerAdamState));

    //Setup params
    params->beta_1 = beta_1;
    params->beta_2 = beta_2;
    
    //Setup state
    state->model =  model;
    state->layers = model->num_layers;
    state->weight_m_t = (Matrix **)malloc(sizeof(Matrix *) * state->layers);
    state->weight_v_t = (Matrix **)malloc(sizeof(Matrix *) * state->layers);
    state->bias_m_t = (Vector **)malloc(sizeof(Vector *) * state->layers);
    state->bias_v_t = (Vector **)malloc(sizeof(Vector *) * state->layers);

    for (size_t i = 0; i < state->layers; i ++) {

        Layer *cur_layer = model->layers[i];

        state->weight_m_t[i] = create_matrix(cur_layer->weights->rows, cur_layer->weights->cols);
        state->weight_v_t[i] = create_matrix(cur_layer->weights->rows, cur_layer->weights->cols);

        state->bias_m_t[i] = create_vector(cur_layer->biases->length);
        state->bias_v_t[i] = create_vector(cur_layer->biases->length);

    }

    state->steps = (size_t *)calloc(state->layers, sizeof(size_t));

    o->additional_parameters = (void *) params;
    o->optimizer_state = (void *) state;
    o->destruction = destroy_ADAM_optimizer;

    return o;

}

void destroy_optimizer(Optimizer *o) {

    if (o->destruction != NULL) {
        o->destruction(o);
    }

    free(o);

}