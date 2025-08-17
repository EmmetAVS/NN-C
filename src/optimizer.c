#include "optimizer.h"
#include "model.h"
#include <stdlib.h>

static void optimizer_step_adam(Optimizer *o, Layer *layer, LayerGradients *grads) {

    AdamOptimizer *a_opt = (AdamOptimizer *)o;

    int layer_index = -1;

    for (size_t i = 0; i < a_opt->layers; i ++) {

        if (a_opt->model->layers[i] == layer) {
            layer_index = (int) i;
            break;
        }

    }

    if (layer_index == -1) {
        return;
    }

    a_opt->steps[layer_index] += 1;

    const BASE_TYPE beta_1_pow = pow(a_opt->beta_1, a_opt->steps[layer_index]);
    const BASE_TYPE beta_2_pow = pow(a_opt->beta_2, a_opt->steps[layer_index]);

    //Update weights
    for (size_t output_index = 0; output_index < layer->output_size; output_index ++) {

        for (size_t input_index = 0; input_index < layer->input_size; input_index ++) {

            const size_t index = (output_index * layer->input_size) + input_index;

            BASE_TYPE grad = grads->d_weights->data[index];

            BASE_TYPE m_t = (a_opt->beta_1 * a_opt->weight_m_t[layer_index]->data[index]) + ((1 - a_opt->beta_1) * grad);
            BASE_TYPE v_t = (a_opt->beta_2 * a_opt->weight_v_t[layer_index]->data[index]) + ((1 - a_opt->beta_2) * (grad * grad));

            a_opt->weight_m_t[layer_index]->data[index] = m_t;
            a_opt->weight_v_t[layer_index]->data[index] = v_t;

            //Correct bias
            BASE_TYPE no_bias_m_t = (m_t) / (1 - beta_1_pow);
            BASE_TYPE no_bias_v_t = (v_t) / (1 - beta_2_pow);

            layer->weights->data[index] -= o->learning_rate * (no_bias_m_t / (sqrt(no_bias_v_t) + EPSILON));

        }

    }

    //Update biases
    for (size_t output_index = 0; output_index < layer->output_size; output_index ++) {

        BASE_TYPE grad = grads->d_biases->data[output_index];

        BASE_TYPE m_t = (a_opt->beta_1 * a_opt->bias_m_t[layer_index]->data[output_index]) + ((1 - a_opt->beta_1) * grad);
        BASE_TYPE v_t = (a_opt->beta_2 * a_opt->bias_v_t[layer_index]->data[output_index]) + ((1 - a_opt->beta_2) * (grad * grad));

        a_opt->bias_m_t[layer_index]->data[output_index] = m_t;
        a_opt->bias_v_t[layer_index]->data[output_index] = v_t;

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
    o->destruction = NULL;

    return o;
}

static void destroy_ADAM_optimizer(Optimizer *o) {
    AdamOptimizer *a_opt = (AdamOptimizer *)o;

    for (size_t i = 0; i < a_opt->layers; i ++) {

        destroy_matrix(a_opt->weight_m_t[i]);
        destroy_matrix(a_opt->weight_v_t[i]);

        destroy_vector(a_opt->bias_m_t[i]);
        destroy_vector(a_opt->bias_v_t[i]);

    }

    free(a_opt->weight_m_t);
    free(a_opt->weight_v_t);
    free(a_opt->bias_m_t);
    free(a_opt->bias_v_t);
    free(a_opt->steps);

}

Optimizer *create_ADAM_optimizer(Model *model, BASE_TYPE learning_rate, BASE_TYPE beta_1, BASE_TYPE beta_2) {

    AdamOptimizer* o = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
    ((Optimizer *) o)->learning_rate = learning_rate;
    ((Optimizer *) o)->step = optimizer_step_adam;

    //Setup params
    o->beta_1 = beta_1;
    o->beta_2 = beta_2;
    
    //Setup state
    o->model =  model;
    o->layers = model->num_layers;
    o->weight_m_t = (Matrix **)malloc(sizeof(Matrix *) * o->layers);
    o->weight_v_t = (Matrix **)malloc(sizeof(Matrix *) * o->layers);
    o->bias_m_t = (Vector **)malloc(sizeof(Vector *) * o->layers);
    o->bias_v_t = (Vector **)malloc(sizeof(Vector *) * o->layers);

    for (size_t i = 0; i < o->layers; i ++) {

        Layer *cur_layer = model->layers[i];

        o->weight_m_t[i] = create_matrix(cur_layer->weights->rows, cur_layer->weights->cols);
        o->weight_v_t[i] = create_matrix(cur_layer->weights->rows, cur_layer->weights->cols);

        o->bias_m_t[i] = create_vector(cur_layer->biases->length);
        o->bias_v_t[i] = create_vector(cur_layer->biases->length);

    }

    o->steps = (size_t *)calloc(o->layers, sizeof(size_t));
    ((Optimizer *) o)->destruction = destroy_ADAM_optimizer;

    return (Optimizer *) o;

}

void destroy_optimizer(Optimizer *o) {

    if (o->destruction != NULL) {
        o->destruction(o);
    }

    free(o);

}