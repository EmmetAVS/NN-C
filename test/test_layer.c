#include "layer.h"
#include "activation.h"
#include "tests.h"
#include <assert.h>
#include <stdio.h>

int main() {

    Layer *l = create_layer(10, 5, activation_loss_softmax_cross_entropy);

    assert(l != NULL);

    destroy_layer(l);

    return 0;

}