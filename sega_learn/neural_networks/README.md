# Neural Networks Module

The Neural Networks module in **SEGA_LEARN** is a fully featured framework for building, training, and evaluating custom neural network models for **both classification and regression tasks**. It provides implementations of key neural network components using pure NumPy for clarity and flexibility, as well as Numba-accelerated and CuPy-accelerated versions for performance gains. The module supports a wide range of functionalities including diverse layer types, activation functions, loss calculations, optimizers, learning rate schedulers, and various utility functions for efficient forward and backward propagation.

## How Neural Networks Work

Neural networks are computational models inspired by the human brain. They consist of layers of interconnected nodes (neurons) that process input data to produce an output. The key concepts behind neural networks include:

1. **Layers**: Neural networks are composed of input, hidden, and output layers. Each layer contains neurons that apply transformations to the data.
2. **Weights and Biases**: Each connection between neurons has an associated weight and bias, which are learned during training.
3. **Activation Functions**: Non-linear functions applied to the output of each neuron to introduce non-linearity into the model.
4. **Forward Propagation**: The process of passing input data through the network to generate predictions.
5. **Backward Propagation**: The process of calculating gradients and updating weights to minimize the loss function.

### Classification
For classification tasks, neural networks output probabilities for each class. The class with the highest probability is selected as the prediction.

### Regression
For regression tasks, the network is configured to predict continuous target values. The final layer typically uses a linear activation (or no activation, specified as `'none'` or `'linear'`), and the loss function measures the difference between the predicted and actual continuous values (e.g., Mean Squared Error). Data scaling, especially for the target variable `y`, is often crucial for achieving good performance in regression.

### Advanced Features
- **Dropout**: A regularization technique that randomly drops neurons during training to prevent overfitting.
- **L2 Regularization**: Adds a penalty to the loss function based on the magnitude of the weights to prevent overfitting.
- **Customizable Layers**: Support for dense, convolutional, recurrent, and other types of layers.

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
