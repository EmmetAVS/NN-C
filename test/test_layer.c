#include "layer.h"
#include "activations.h"
#include "tests.h"
#include <assert.h>
#include <stdio.h>

int main() {
    
    nnlib_startup();

    Layer *l = create_layer(10, 5, activation_loss_softmax_cross_entropy);

    assert(l != NULL);

    destroy_layer(l);

    return 0;

}