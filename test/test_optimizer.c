#include "tests.h"
#include "optimizer.h"

void test_create_and_destroy_optimizer() {
    BASE_TYPE lr = 0.01f;

    Optimizer *opt = create_SGD_optimizer(lr);
    assert(opt != NULL);
    assert(opt->learning_rate == lr);

    destroy_optimizer(opt);
}

int main() {
    test_create_and_destroy_optimizer();
    return 0;
}