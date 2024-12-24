## Optimzers

They adjust the learning rate, and the learning is used in backpropagation to modify the weights and biases to reduce loss function.

### Adam

Combines the benefits of momentum and adaptive learning rates.

- Momentum: accelerates gradient descent by adding previous update (gradient) to current one.

- Adaptive learning rate: learning rate is adjusted based on the progress of weight updates. (large gradients get smaller updates and vice versa)

### SGD (Stochastic gradient descent)

- Acts like normal gradient descent, difference is that is introduces randomness, by using batches of training data.


### RMSProp (Root Mean Square Propagation)

- maintains a moving average of the squared gradients, giving more weight to recent gradients.
- avoids rapid learning rate decay.


## Loss function vs Cost function

Measures True - Prediction with a mathematical equation.

- loss function: Measure model error on a single instance
- cost function: Measure model error on a group of data