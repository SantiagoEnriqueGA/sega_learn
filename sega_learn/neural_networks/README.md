# Neural Networks Module

The Neural Networks module in **SEGA_LEARN** is a fully featured framework for building, training, and evaluating custom neural network models for **both classification and regression tasks**. It provides implementations of key neural network components using pure NumPy for clarity and flexibility, as well as Numba-accelerated and CuPy-accelerated versions for performance gains. The module supports a wide range of functionalities including diverse layer types, activation functions, loss calculations, optimizers, learning rate schedulers, and various utility functions for efficient forward and backward propagation.

## How Neural Networks Work

Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process input data to produce an output. The key concepts behind neural networks include:

1. **Layers**: Neural networks are composed of input, hidden, and output layers. Each layer contains neurons that apply transformations to the data.
2. **Weights and Biases**: Each connection between neurons has an associated weight and bias, which are learned during training.
3. **Activation Functions**: Non-linear functions applied to the output of each neuron to introduce non-linearity into the model.
4. **Forward Propagation**: The process of passing input data through the network to generate predictions.
5. **Backward Propagation**: The process of calculating gradients and updating weights to minimize the loss function.
6. **Loss Functions**: Functions that measure the difference between predicted and actual values, guiding the optimization process.
7. **Optimizers**: Algorithms that adjust the weights and biases based on the gradients calculated during backpropagation.

### Classification
For classification tasks, neural networks output probabilities for each class. The class with the highest probability is selected as the prediction.

### Regression
For regression tasks, the network is configured to predict continuous target values. The final layer typically uses a linear activation (or no activation, specified as `'none'` or `'linear'`), and the loss function measures the difference between the predicted and actual continuous values (e.g., Mean Squared Error). Data scaling, especially for the target variable `y`, is often crucial for achieving good performance in regression.

### Advanced Features
- **Dropout**: A regularization technique that randomly drops neurons during training to prevent overfitting.
- **L2 Regularization**: Adds a penalty to the loss function based on the magnitude of the weights to prevent overfitting.
- **Customizable Layers**: Support for dense, convolutional, recurrent, and other types of layers.
- **Learning Rate Schedulers**: Adjust the learning rate during training to improve convergence.

## Backends

The module supports multiple backends for computation:

1. **Base Backend**: A CPU-based implementation for general-purpose use.
2. **Numba Backend**: Uses Numba for Just-In-Time (JIT) compilation to optimize performance.
3. **CuPy Backend**: Leverages GPU acceleration via CuPy for faster computations.

## Algorithms

### Layer Types
Neural networks can utilize various types of layers, including:
- **Dense Layer**: Fully connected layer where each neuron receives input from all neurons in the previous layer.
- **Convolutional Layer**: Specialized layer for processing grid-like data such as images, applying convolution operations.
- **Recurrent Layer**: Designed for sequential data, allowing information to persist across time steps.
- **Flatten Layer**: Converts multi-dimensional input into a one-dimensional array.

### Activation Functions

Activation functions introduce non-linearity into the neural network, enabling it to learn complex patterns in the data. They are applied to the output of each neuron in a layer before passing it to the next layer. Choosing the right activation function is crucial for the performance of the neural network.

#### Common Activation Functions

1. **Sigmoid Function**
   - Formula: $\sigma(x) = \frac{1}{1 + e^{-x}}$
   - Range: (0, 1)
   - Usage: Commonly used in the output layer for binary classification tasks.

2. **Tanh Function**
   - Formula: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
   - Range: (-1, 1)
   - Usage: Often used in hidden layers to center the data around zero, which can speed up convergence.

3. **ReLU (Rectified Linear Unit)**
   - Formula: $f(x) = \max(0, x)$
   - Range: [0, ∞)
   - Usage: The most commonly used activation function in hidden layers due to its simplicity and efficiency.

4. **Leaky ReLU**
   - Formula: $f(x) = x$ if $x > 0$, else $f(x) = \alpha x$
   - Range: (-∞, ∞)
   - Usage: A variant of ReLU that allows a small gradient for negative values to prevent dead neurons.

5. **Softmax Function**
   - Formula: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$
   - Range: (0, 1), with the sum of outputs equal to 1.
   - Usage: Used in the output layer for multi-class classification tasks.

6. **Linear / None:**
   - For regression tasks, the output layer typically uses no activation (equivalent to a linear activation) to allow for predicting unbounded continuous values.
   - Specify `'linear'` or `'none'` as the activation for the final layer in regression networks.

#### Choosing an Activation Function
- **Sigmoid**: Use for binary classification or probabilistic outputs.
- **Tanh**: Use in hidden layers when data needs to be centered around zero.
- **ReLU**: Default choice for hidden layers in most neural networks.
- **Leaky ReLU**: Use when ReLU leads to dead neurons.
- **Softmax**: Use in the output layer for multi-class classification.


---

### Loss Functions

Loss functions are critical in training neural networks as they quantify the difference between the predicted outputs and the actual targets. The choice of loss function can significantly impact the performance of the model. Here are the loss functions implemented in the module:

1. **Cross Entropy Loss (Categorical Crossentropy)**:
   - Used for multi-class classification problems.
   - Formula: $` \text{Loss} = -\frac{1}{m} \sum \left(y \cdot \log(p + \epsilon)\right) `$
   - Where:
     - $` y `$ is the one-hot encoded target vector.
     - $` p `$ is the predicted probability vector (output of softmax).
     - $` m `$ is the number of samples.
     - $` \epsilon `$ is a small constant to prevent log(0).
   - This loss function penalizes incorrect classifications more heavily, making it suitable for multi-class problems.

2. **Binary Cross Entropy Loss with Logits**:
   - Used for binary classification problems.
   - Formula: $` \text{Loss} = -\frac{1}{m} \sum \left(y \cdot \log(p + \epsilon) + (1-y) \cdot \log(1-p + \epsilon)\right) `$
   - Where:
     - $` y `$ is the binary target vector (0 or 1).
     - $` p `$ is the predicted probability (output of sigmoid).
     - $` m `$ is the number of samples.
     - $` \epsilon `$ is a small constant to prevent log(0).
   - This loss function is suitable for binary classification tasks.

3. **Mean Squared Error (MSE)**:
   - Used for regression problems.
   - Formula: $` \text{Loss} = \frac{1}{m} \sum (y - \hat{y})^2 `$
   - Where:
     - $` y `$ is the actual target value.
     - $` \hat{y} `$ is the predicted value.
     - $` m `$ is the number of samples.
   - This loss function measures the average squared difference between predicted and actual values.

4. **Mean Absolute Error (MAE)**:
   - Another loss function for regression problems.
   - Formula: $` \text{Loss} = \frac{1}{m} \sum |y - \hat{y}| `$
   - Where:
     - $` y `$ is the actual target value.
     - $` \hat{y} `$ is the predicted value.
     - $` m `$ is the number of samples.
   - This loss function measures the average absolute difference between predicted and actual values.

5. **Huber Loss**:
   - A combination of MSE and MAE, robust to outliers.
   - Formula: $` \text{Loss} = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\ \delta \cdot (|y - \hat{y}| - \frac{1}{2}\delta) & \text{otherwise} \end{cases} `$
   - Where:
     - $` y `$ is the actual target value.
     - $` \hat{y} `$ is the predicted value.
     - $` \delta `$ is a threshold parameter.
   - This loss function is less sensitive to outliers than MSE.

---

### Optimizers

Optimizers are algorithms used to adjust the weights and biases of a neural network during training. They aim to minimize the loss function by updating the parameters based on the gradients calculated during backpropagation. Here are the optimizers implemented in the module:

1. **Adam Optimizer**:
   - Combines the benefits of AdaGrad and RMSProp.
   - Maintains a moving average of both the gradients and the squared gradients.
   - Adaptive learning rates for each parameter.
   - Formula:
     - $` m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t `$ (first moment estimate)
     - $` v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 `$ (second moment estimate)
     - $` \hat{m}_t = \frac{m_t}{1 - \beta_1^t} `$ (bias-corrected first moment)
     - $` \hat{v}_t = \frac{v_t}{1 - \beta_2^t} `$ (bias-corrected second moment)
     - $` \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t `$ (parameter update)
   - Where:
     - $` g_t `$ is the gradient at time step $` t `$.
     - $` \beta_1, \beta_2 `$ are decay rates (typically set to 0.9 and 0.999).
     - $` \eta `$ is the learning rate.
     - $` \epsilon `$ is a small constant to prevent division by zero.
     - $` \theta `$ are the parameters (weights and biases).

2. **Adadelta Optimizer**:
   - An extension of AdaGrad that adapts the learning rate based on a moving window of gradient updates.
   - Formula:
     - $` \rho g_t^2 + (1 - \rho) \Delta \theta_{t-1}^2 = g_t^2 `$ (accumulated squared gradients)
     - $` \Delta \theta_t = -\frac{\sqrt{\Delta \theta_{t-1}^2 + \epsilon}}{\sqrt{g_t^2 + \epsilon}} g_t `$ (parameter update)
   - Where:
     - $` \rho `$ is a decay rate (typically set to 0.95).
     - $` \epsilon `$ is a small constant to prevent division by zero.

3. **SGD (Stochastic Gradient Descent)**:
   - A simple and widely used optimizer.
   - Updates parameters based on the average gradient of the mini-batch.
   - Formula: $` \theta_{t+1} = \theta_t - \eta \cdot \nabla L(\theta_t) `$
   - Where:
     - $` \nabla L(\theta_t) `$ is the gradient of the loss function with respect to the parameters at time step $` t `$.
     - $` \eta `$ is the learning rate.


---

### Learning Rate Schedulers

Learning rate schedulers adjust the learning rate during training to improve convergence and performance. Scheduling the learning rate can help the model escape local minima and improve generalization. They can be set to execute at specific intervals or based on certain conditions, the `Plateau Scheduler` allows for dynamic adjustments based on validation loss. Here are the learning rate schedulers implemented in the module:

1. **Step Scheduler**:
   - Reduces the learning rate by a fixed factor every set number of epochs.
   - Formula: $` \text{new\_lr} = \text{initial\_lr} \times \text{factor}^{\text{epoch} // \text{step}} `$
   - Where:
     - $` \text{initial\_lr} `$ is the initial learning rate.
     - $` \text{factor} `$ is the reduction factor.
     - $` \text{epoch} `$ is the current epoch number.
     - $` \text{step} `$ is the number of epochs after which to reduce the learning rate.

2. **Exponential Scheduler**:
   - Applies an exponential decay to the learning rate at defined intervals.
   - Formula: $` \text{new\_lr} = \text{initial\_lr} \times e^{-\text{decay\_rate} \times \text{epoch}} `$
   - Where:
     - $` \text{initial\_lr} `$ is the initial learning rate.
     - $` \text{decay\_rate} `$ is the decay rate.
     - $` \text{epoch} `$ is the current epoch number.

3. **Plateau Scheduler**:
   - Monitors the loss and reduces the learning rate when improvements plateau.
   - Formula: $` \text{new\_lr} = \text{initial\_lr} \times \text{factor} `$
   - Where:
     - $` \text{initial\_lr} `$ is the initial learning rate.
     - $` \text{factor} `$ is the reduction factor.
     - The scheduler checks the validation loss at regular intervals and reduces the learning rate if no improvement is observed.


## Examples

### Base Backend Example
```python
from sega_learn.neural_networks import BaseBackendNeuralNetwork
import numpy as np

# Generate sample data
X_train = np.random.rand(100, 64)
y_train = np.random.randint(0, 10, size=100)

# Initialize and train the neural network
nn = BaseBackendNeuralNetwork(layers=[64, 32, 10], dropout_rate=0.2, reg_lambda=0.01)
nn.train(X_train, y_train, epochs=50, batch_size=16)

# Predict on new data
predictions = nn.predict(X_train)
print(predictions)
```

### Numba Backend Example
```python
from sega_learn.neural_networks import NumbaBackendNeuralNetwork
import numpy as np

# Generate sample data
X_train = np.random.rand(100, 64)
y_train = np.random.randint(0, 10, size=100)

# Initialize and train the neural network
nn = NumbaBackendNeuralNetwork(layers=[64, 32, 10], compile_numba=True)
nn.train(X_train, y_train, epochs=50, batch_size=16)

# Predict on new data
predictions = nn.predict(X_train)
print(predictions)
```

### CuPy Backend Example
```python
from sega_learn.neural_networks import CuPyBackendNeuralNetwork
import numpy as np

# Generate sample data
X_train = np.random.rand(100, 64)
y_train = np.random.randint(0, 10, size=100)

# Initialize and train the neural network
nn = CuPyBackendNeuralNetwork(layers=[64, 32, 10], dropout_rate=0.2, reg_lambda=0.01)
nn.train(X_train, y_train, epochs=50, batch_size=16)

# Predict on new data
predictions = nn.predict(X_train)
print(predictions)
```

### Regression Example
```python
from sega_learn.neural_networks import BaseBackendNeuralNetwork
import numpy as np

# Generate sample data
X_train = np.random.rand(100, 64)
y_train = np.random.rand(100, 1)

# Initialize and train the neural network
nn = BaseBackendNeuralNetwork(layers=[64, 32, 1], dropout_rate=0.2, reg_lambda=0.01, regression=True)
nn.train(X_train, y_train, epochs=50, batch_size=16)

# Predict on new data
predictions = nn.predict(X_train)
print(predictions)
```
