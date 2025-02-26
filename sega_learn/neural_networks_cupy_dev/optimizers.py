import cupy as cp

class AdamOptimizer:
    """
    Adam optimizer class for training neural networks.
    Formula: w = w - alpha * m_hat / (sqrt(v_hat) + epsilon) - lambda * w 
    Derived from: https://arxiv.org/abs/1412.6980
    """
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, reg_lambda=0.01):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.state = {}  # Dictionary to hold state per layer: {'m': ..., 'v': ...}
        self.t = 0       # Global time step

    def initialize(self, layers):
        for layer in layers:
            self.state[id(layer)] = {
                'm': cp.zeros_like(layer.weights),
                'v': cp.zeros_like(layer.weights)
            }

    def update(self, layer, dW, db):
        # Ensure gradients are CuPy arrays
        if not hasattr(dW, 'get'):
            dW_gpu = cp.asarray(dW)
        else:
            dW_gpu = dW
        if not hasattr(db, 'get'):
            db_gpu = cp.asarray(db)
        else:
            db_gpu = db

        self.t += 1
        state = self.state[id(layer)]
        state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * dW_gpu
        state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * cp.square(dW_gpu)

        m_hat = state['m'] / (1 - self.beta1 ** self.t)
        v_hat = state['v'] / (1 - self.beta2 ** self.t)

        # Ensure weights and biases are CuPy arrays
        if not hasattr(layer.weights, 'get'):
            layer.weights = cp.asarray(layer.weights)
        if not hasattr(layer.biases, 'get'):
            layer.biases = cp.asarray(layer.biases)

        layer.weights -= self.learning_rate * (m_hat / (cp.sqrt(v_hat) + self.epsilon) + self.reg_lambda * layer.weights)
        layer.biases -= self.learning_rate * db_gpu

class SGDOptimizer:
    """
    Stochastic Gradient Descent (SGD) optimizer class for training neural networks.
    Formula: w = w - learning_rate * dW, b = b - learning_rate * db
    """
    def __init__(self, learning_rate=0.001, momentum=0.0, reg_lambda=0.0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.reg_lambda = reg_lambda
        self.state = {}  # Dictionary to hold state per layer: {'velocity': ...}

    def initialize(self, layers):
        for layer in layers:
            self.state[id(layer)] = {
                'velocity': cp.zeros_like(layer.weights)
            }

    def update(self, layer, dW, db):
        if not hasattr(dW, 'get'):
            dW_gpu = cp.asarray(dW)
        else:
            dW_gpu = dW
        if not hasattr(db, 'get'):
            db_gpu = cp.asarray(db)
        else:
            db_gpu = db

        if not hasattr(layer.weights, 'get'):
            layer.weights = cp.asarray(layer.weights)
        if not hasattr(layer.biases, 'get'):
            layer.biases = cp.asarray(layer.biases)

        state = self.state[id(layer)]
        state['velocity'] = self.momentum * state['velocity'] - self.learning_rate * dW_gpu
        layer.weights += state['velocity'] - self.learning_rate * self.reg_lambda * layer.weights
        layer.biases -= self.learning_rate * db_gpu

class AdadeltaOptimizer:
    """
    Adadelta optimizer class for training neural networks.
    Formula: 
        E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g^2
        Delta_x = - (sqrt(E[delta_x^2]_{t-1} + epsilon) / sqrt(E[g^2]_t + epsilon)) * g
        E[delta_x^2]_t = rho * E[delta_x^2]_{t-1} + (1 - rho) * Delta_x^2
    Derived from: https://arxiv.org/abs/1212.5701
    """
    def __init__(self, learning_rate=1.0, rho=0.95, epsilon=1e-6, reg_lambda=0.0):
        self.learning_rate = learning_rate
        self.rho = rho
        self.epsilon = epsilon
        self.reg_lambda = reg_lambda
        self.state = {}  # Dictionary to hold state per layer: {'E_g2': ..., 'E_delta_x2': ...}

    def initialize(self, layers):
        for layer in layers:
            self.state[id(layer)] = {
                'E_g2': cp.zeros_like(layer.weights),
                'E_delta_x2': cp.zeros_like(layer.weights)
            }

    def update(self, layer, dW, db):
        if not hasattr(dW, 'get'):
            dW_gpu = cp.asarray(dW)
        else:
            dW_gpu = dW
        if not hasattr(db, 'get'):
            db_gpu = cp.asarray(db)
        else:
            db_gpu = db

        if not hasattr(layer.weights, 'get'):
            layer.weights = cp.asarray(layer.weights)
        if not hasattr(layer.biases, 'get'):
            layer.biases = cp.asarray(layer.biases)

        state = self.state[id(layer)]
        state['E_g2'] = self.rho * state['E_g2'] + (1 - self.rho) * cp.square(dW_gpu)
        delta_x = - (cp.sqrt(state['E_delta_x2'] + self.epsilon) / cp.sqrt(state['E_g2'] + self.epsilon)) * dW_gpu
        state['E_delta_x2'] = self.rho * state['E_delta_x2'] + (1 - self.rho) * cp.square(delta_x)
        layer.weights += delta_x - self.learning_rate * self.reg_lambda * layer.weights
        layer.biases -= self.learning_rate * db_gpu
