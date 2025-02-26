import cupy as cp

class CrossEntropyLoss:
    """
    Custom cross entropy loss implementation using cupy for multi-class classification.
    Formula: -sum(y * log(p) + (1 - y) * log(1 - p)) / m
    Methods:
        __call__(self, logits, targets): Calculate the cross entropy loss.
    """
    def __call__(self, logits, targets):
        """
        Calculate the cross entropy loss.
        Args:
            logits (cp.ndarray): The logits (predicted values) of shape (num_samples, num_classes).
            targets (cp.ndarray): The target labels of shape (num_samples, num_classes).
        Returns:
            float: The cross entropy loss.
        """
        # Convert numpy arrays to cupy if needed
        if not hasattr(logits, 'get'):  logits = cp.asarray(logits)
        if not hasattr(targets, 'get'): targets = cp.asarray(targets)
        
        # One-hot encode targets if they are not already
        if targets.ndim == 1:
            targets = cp.eye(logits.shape[1])[targets]

        exp_logits = cp.exp(logits - cp.max(logits, axis=1, keepdims=True)) # Exponential of logits, subtract max to prevent overflow
        probs = exp_logits / cp.sum(exp_logits, axis=1, keepdims=True)      # Probabilities from logits
        loss = -cp.sum(targets * cp.log(probs + 1e-15)) / logits.shape[0]   # Cross-entropy loss
        
        # Convert result to scalar
        return float(loss.get()) if hasattr(loss, 'get') else float(loss)

class BCEWithLogitsLoss:
    """
    Custom binary cross entropy loss with logits implementation using cupy.
    Formula: -mean(y * log(p) + (1 - y) * log(1 - p))
    Methods:
        __call__(self, logits, targets): Calculate the binary cross entropy loss.
    """
    def __call__(self, logits, targets):
        """
        Calculate the binary cross entropy loss.
        Args:
            logits (cp.ndarray): The logits (predicted values) of shape (num_samples,).
            targets (cp.ndarray): The target labels of shape (num_samples,).
        Returns:
            float: The binary cross entropy loss.
        """
        # Convert numpy arrays to cupy if needed
        if not hasattr(logits, 'get'):
            logits = cp.asarray(logits)
        if not hasattr(targets, 'get'):
            targets = cp.asarray(targets)
            
        probs = 1 / (1 + cp.exp(-logits))                                                               # Apply sigmoid to logits to get probabilities
        loss = -cp.mean(targets * cp.log(probs + 1e-15) + (1 - targets) * cp.log(1 - probs + 1e-15))    # Binary cross-entropy loss
        
        # Convert result to scalar
        return float(loss.get()) if hasattr(loss, 'get') else float(loss)