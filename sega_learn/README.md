# Neural Networks Module

The Neural Networks Module in **SEGA_LEARN** is a fully featured framework for building, training, and evaluating custom neural network models. It provides implementations of key neural network components using pure NumPy for clarity and flexibility as well as Numba-accelerated versions for performance gains. The module supports a wide range of functionalities including diverse layer types, activation functions, loss calculations, optimizers, learning rate schedulers, and various utility functions for efficient forward and backward propagation.

---

## Conceptual Overview

Neural networks are a class of machine learning models inspired by the human brain’s architecture. In this module, a neural network is constructed as a series of layers where each layer performs a linear transformation followed by a non-linear activation. Training is achieved via backpropagation—a method that calculates gradients for each parameter and updates them using an optimizer. This module supports both single-threaded NumPy implementations and Numba-accelerated routines for increased computational performance.

---

## Architecture

- **Layers & Weight Initialization:**
  The module provides a `Layer` class (and its Numba counterpart `JITLayer`) that encapsulates the weights, biases, and activation functions for a network layer. Weight initialization uses He initialization for ReLU and Leaky ReLU activations and a scaled approach for others.

- **Forward & Backward Propagation:**
  The `BaseBackendNeuralNetwork` and `NumbaBackendNeuralNetwork` classes orchestrate the forward pass, computing activations for each layer (with optional dropout during training), and the backward pass, where gradients are calculated layer by layer. Utility functions (in `numba_utils.py`) support these operations with Numba-compiled versions for faster computations.

- **Dual Backend Support:**
  Users can choose between a pure NumPy backend (`BaseBackendNeuralNetwork`) or a Numba-accelerated version (`NumbaBackendNeuralNetwork`) by selecting the appropriate class. This design allows for both ease of debugging and high-performance training.

---

## Activation Functions

The module provides a robust set of activation functions along with their derivatives for backpropagation:

- **ReLU (Rectified Linear Unit):**
  $` \text{ReLU}(z) = \max(0, z) `$
  Derivative: $` f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases} `$

- **Leaky ReLU:**
  $` \text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases} `$
  Derivative: $` f'(z) = \begin{cases} 1 & z > 0 \\ \alpha & z \leq 0 \end{cases} `$

- **Tanh:**
  $` \tanh(z) `$ maps inputs to the range $`[-1, 1] `$.

- **Sigmoid:**
  $` \sigma(z) = \frac{1}{1+\exp(-z)} `$ maps inputs to $`[0, 1] `$.

- **Softmax:**
  Converts logits into a probability distribution across classes:
  $` \text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)} `$

These functions are implemented in both standard (in `activations.py`) and Numba-compatible forms (in `numba_utils.py` and used by `JITLayer`).

---

## Layers

- **Standard Layers:**
  The `Layer` class (in `layers.py`) sets up a single layer with its weight matrix, bias vector, and chosen activation function. It includes methods to reset gradients, apply the activation function, and compute the derivative during backpropagation.

- **Numba-Accelerated Layers:**
  For performance-critical applications, `JITLayer` (in `layers_jit.py`) uses Numba’s jitclass decorator to compile layer operations, including weight initialization, activation, and gradient resetting.

---

## Loss Functions

Two primary loss functions are provided:

- **CrossEntropyLoss:**
  Used for multi-class classification, it computes the loss as:
  $` \text{Loss} = -\frac{1}{m} \sum \left(y \cdot \log(p + \epsilon)\right) `$
  where targets are one-hot encoded and $` p `$ are the probabilities obtained via softmax.

- **BCEWithLogitsLoss:**
  Designed for binary classification, combining a sigmoid activation with binary cross-entropy:
  $` \text{Loss} = -\frac{1}{m} \sum \left(y \cdot \log(p + \epsilon) + (1-y) \cdot \log(1-p + \epsilon)\right) `$

Both standard implementations (in `loss.py`) and Numba-compiled versions (in `loss_jit.py`) are included.

---

## Optimizers

The module supports several optimizers, each with its own update formula:

- **AdamOptimizer:**
  Combines momentum and adaptive learning rates using first and second moment estimates.
  Update rule:
  $` w = w - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon} - \lambda w `$
  Standard implementation is in `optimizers.py`, while `JITAdamOptimizer` (in `optimizers_jit.py`) offers a Numba-accelerated alternative.

- **SGDOptimizer:**
  Implements basic stochastic gradient descent, optionally with momentum:
  $` w = w - \text{learning rate} \times dW - \lambda w, \quad b = b - \text{learning rate} \times db `$
  Also available in a Numba version as `JITSGDOptimizer`.

- **AdadeltaOptimizer:**
  Adjusts learning rates based on a moving window of gradient updates:
  $` E[g^2]_t = \rho E[g^2]_{t-1} + (1 - \rho) g^2 `$

  $` \Delta x = - \frac{\sqrt{E[\Delta x^2]_{t-1} + \epsilon}}{\sqrt{E[g^2]_t + \epsilon}} g `$

  $` E[\Delta x^2]_t = \rho E[\Delta x^2]_{t-1} + (1 - \rho) \Delta x^2 `$
  Implemented in both standard (`optimizers.py`) and Numba (`optimizers_jit.py`) versions.

---

## Learning Rate Schedulers

The module provides multiple strategies to adjust the learning rate during training:

- **Step Scheduler (`lr_scheduler_step`):**
  Reduces the learning rate by a fixed factor every set number of epochs.

- **Exponential Scheduler (`lr_scheduler_exp`):**
  Applies an exponential decay to the learning rate at defined intervals.

- **Plateau Scheduler (`lr_scheduler_plateau`):**
  Monitors the loss and reduces the learning rate when improvements plateau.

All scheduler classes are defined in `schedulers.py` with user-friendly messages to track adjustments.

---

## Utility Functions (Numba Utils)

The `numba_utils.py` module contains helper functions that accelerate various operations via Numba, including:

- **Forward & Backward Passes:**
  Functions such as `forward_jit` and `backward_jit` manage data propagation and gradient computation in a highly optimized manner.

- **Activation Functions:**
  Numba-compatible versions of ReLU, Leaky ReLU, Tanh, Sigmoid, and Softmax are provided for rapid computation.

- **Regularization & Dropout:**
  Functions to compute L2 regularization and apply dropout (e.g., `apply_dropout_jit`) are also included.

- **Batch Processing:**
  The `process_batches` function allows for efficient mini-batch training with parallel processing support.

---

## Neural Network Classes

The core classes, `BaseBackendNeuralNetwork` and `NumbaBackendNeuralNetwork`, integrate all the above components:

- **Model Construction:**
  Accepts a list of layer sizes and optionally a list of activation functions. It builds the network by stacking layers (using either the standard or Numba-accelerated version).

- **Forward Propagation:**
  Computes predictions by propagating input data through the layers. Dropout is applied during training to help regularize the model.

- **Backward Propagation:**
  Calculates gradients for all parameters using the chain rule and updates them via the chosen optimizer.

- **Training Methods:**
  Provides a `train` method for standard training and a `train_numba` method for accelerated training. It supports mini-batch gradient descent, parallel batch processing (via joblib), early stopping, and integration with learning rate schedulers.

- **Evaluation:**
  Methods to compute loss and accuracy on training or validation data are included.

---

## Example Usage

```python
from sega_learn.neural_networks import BaseBackendNeuralNetwork, AdamOptimizer, lr_scheduler_step
import numpy as np

# Generate sample data
X_train = np.random.rand(100, 10)
y_train = np.random.randint(0, 2, size=(100,))

# Initialize the neural network (using standard NumPy backend)
nn = BaseBackendNeuralNetwork(layer_sizes=[10, 5, 1], dropout_rate=0.2, reg_lambda=0.01)

# Set up an optimizer
optimizer = AdamOptimizer(learning_rate=0.001)

# Optionally, set up a learning rate scheduler
scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)

# Train the neural network
nn.train(X_train, y_train, optimizer=optimizer, epochs=100, batch_size=32, lr_scheduler=scheduler)

# Evaluate the neural network
accuracy, predictions = nn.evaluate(X_train, y_train)
print(f"Training Accuracy: {accuracy:.4f}")
```
