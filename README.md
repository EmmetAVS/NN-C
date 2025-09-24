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
   - SGD (params: learning_rate)
   - ADAM (params: learning_rate, beta_1, beta_2)
   - Support for customs
6. Full training loop (as demonstrated in test/test_training_loop.c)
   ```
   Model *model = ...;
   Optimizer *opt = ...;

   for (int epoch = 0; epoch < epochs; epoch ++) {

      for (int batch = 0; batch < batches; batch ++) {
      
         model_zero_grads(model);
         model_set_max_grads(model, batch_size);
      
         for (int i = 0; i < batch_size; ++i) {
            Vector *output = model_forward(model, inputs[batch][i]);
            model_backward(model, labels[batch][i]);
            destroy_vector(output);
         }
      
         model_average_grads(model);
         model_step(model, opt);
   
      }

   }
   ```
7. Seralization of Model Parameters
8. Utils including Shuffler to shuffle data of any size/type, argmax, one hot encoding
9. Data loading from CSV's (Only supports loading data as either int, string (char * with '\0' termination), or float
   - Full use of all features can be observed in examples/digit_recognition/main.c

## Notes
- **Not optimized, no GPU usage**
- **Still continuing to work on adding additional/more complex features and model types, along with examples**
- **Written just for me to learn, implementing basic features as I learnt about them in [3Blue1Brown's videos](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)**
