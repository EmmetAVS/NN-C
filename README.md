# Neural Networks Framework
- **Written in pure C, no external libraries**
## Features
1. Activation Functions
   - ReLU
   - Sigmoid
   - Softmax
   - Support for customs
2. Loss Functions
   - Mean Squared Error
   - Cross Entropy Loss
   - Softmax Activation + Cross Entropy Loss combination
   - Support for customs
3. Gradient Descent, Backpropogation
4. Optimizers
   - SGD with customizable learning rate
6. Full training loop (as demonstrated in test/test_training_loop.c)
   ```
   Model *model = ...;
   Optimizer *opt = ...;
   
   model_zero_grads(model);
   model_set_max_grads(model, batch_size);

   for (int i = 0; i < samples; ++i) {
      model_forward(model, inputs[i]);
      model_backward(model, labels[i]);
   }

   model_average_grads(model);
   model_step(model, opt);
   ```

## Notes
- **Not optimized, no GPU usage**
- **Does have memory leaks in the larger training loop**
- **Still continuing to work on fixing memory leaks and adding additional/more complex optimizers/features, along with examples**
- **Written just for me to learn, implementing features as I learnt about them in [3Blue1Brown's videos](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)**
