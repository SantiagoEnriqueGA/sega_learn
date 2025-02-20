# Neural Networks Module

The neural networks module in SEGA_LEARN provides implementations of various neural network architectures and optimization algorithms. These tools are designed to facilitate the training and evaluation of custom neural network models. The module includes classes for building neural networks, defining loss functions, and selecting optimization algorithms.

## Neural Networks Conceptual Overview

### Overview
Neural networks are a class of machine learning models inspired by the structure and function of the human brain. They consist of layers of interconnected nodes (neurons) that process input data and learn to make predictions or decisions based on that data.

### Architecture
A typical neural network consists of an input layer, one or more hidden layers, and an output layer. Each layer contains a certain number of neurons, and each neuron is connected to neurons in the previous and next layers through weighted connections.

### Activation Functions
Activation functions introduce non-linearity into the network, allowing it to learn complex patterns. Common activation functions include ReLU, sigmoid, and softmax.

### Training
Training a neural network involves adjusting the weights of the connections between neurons to minimize the difference between the predicted and actual outputs. This is done using optimization algorithms such as gradient descent and its variants (e.g., Adam, SGD).

### Backpropagation
Backpropagation is the process of calculating the gradient of the loss function with respect to each weight by applying the chain rule. This gradient is then used to update the weights during training.

## Optimizers

### AdamOptimizer
Adam (Adaptive Moment Estimation) is an optimization algorithm that computes adaptive learning rates for each parameter. It combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp.
v
#### Formula
$` w = w - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} - \lambda w `$

#### Usage
```python
from sega_learn.neural_networks.optimizers import AdamOptimizer

# Initialize the Adam optimizer
adam_optimizer = AdamOptimizer(learning_rate=0.001)

# Use the optimizer in training
neural_network.train(X_train, y_train, optimizer=adam_optimizer)
```

### SGDOptimizer
Stochastic Gradient Descent (SGD) is an optimization algorithm that updates the model parameters using the gradient of the loss function with respect to the parameters.

#### Formula
$` w = w - \text{learning rate} \cdot dW - \lambda \cdot w `$  
$` b = b - \text{learning rate} \cdot db `$

#### Usage
```python
from sega_learn.neural_networks.optimizers import SGDOptimizer

# Initialize the SGD optimizer
sgd_optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9)

# Use the optimizer in training
neural_network.train(X_train, y_train, optimizer=sgd_optimizer)
```

### AdadeltaOptimizer
Adadelta is an optimization algorithm that adapts learning rates based on a moving window of gradient updates, instead of accumulating all past gradients.

#### Formula
**Gradient Accumulations:**
$` E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g^2 `$  

**Parameter Updates:**
$` \Delta x = - \frac{\sqrt{E[\Delta x^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g `$  

**Update Accumulations:**
$` E[\Delta x^2]_t = \rho E[\Delta x^2]_{t-1} + (1 - \rho) \Delta x^2 `$  

#### Usage
```python
from sega_learn.neural_networks.optimizers import AdadeltaOptimizer

# Initialize the Adadelta optimizer
adadelta_optimizer = AdadeltaOptimizer(learning_rate=1.0)

# Use the optimizer in training
neural_network.train(X_train, y_train, optimizer=adadelta_optimizer)
```

## Loss Functions

### CrossEntropyLoss
Cross entropy loss is used for multi-class classification problems. It measures the performance of a classification model whose output is a probability value between 0 and 1.

#### Formula
$` \text{Loss} = -\sum(y \log(p) + (1 - y) \log(1 - p)) / m `$

#### Usage
```python
from sega_learn.neural_networks.loss import CrossEntropyLoss

# Initialize the loss function
loss_fn = CrossEntropyLoss()

# Calculate the loss
loss = loss_fn(logits, targets)
```

### BCEWithLogitsLoss
Binary Cross Entropy with Logits Loss is used for binary classification problems. It combines a sigmoid layer and the binary cross-entropy loss in one single class.

#### Formula
$` \text{Loss} = -\text{mean}(y \log(p) + (1 - y) \log(1 - p)) `$

#### Usage
```python
from sega_learn.neural_networks.loss import BCEWithLogitsLoss

# Initialize the loss function
loss_fn = BCEWithLogitsLoss()

# Calculate the loss
loss = loss_fn(logits, targets)
```

## Examples

### Neural Network Example
```python
from sega_learn.neural_networks.neuralNetwork import NeuralNetwork
from sega_learn.neural_networks.optimizers import AdamOptimizer
import numpy as np

# Generate sample data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100, 1))

# Initialize the neural network
nn = NeuralNetwork(layer_sizes=[10, 5, 1], dropout_rate=0.2, reg_lambda=0.01)

# Initialize the optimizer
optimizer = AdamOptimizer(learning_rate=0.001)

# Train the neural network
nn.train(X_train, y_train, optimizer=optimizer, epochs=100, batch_size=32)

# Evaluate the neural network
accuracy, predicted = nn.evaluate(X_train, y_train)
print(f"Accuracy: {accuracy}")
```
