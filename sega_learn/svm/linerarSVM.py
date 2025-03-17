from .baseSVM import BaseSVM
import numpy as np

class LinearSVC(BaseSVM):
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01):
        super().__init__(C, tol, max_iter, learning_rate)
        
    def _fit(self, X, y):
        """
        Implement the fitting procedure for LinearSVC.
        
        Algorithm:
            Initialize Parameters: Initialize the weight vector w and bias b.
            Set Hyperparameters: Define the learning rate and the number of iterations.
            Gradient Descent Loop: Iterate over the dataset to update the weights and bias using gradient descent.
            Compute Hinge Loss: Calculate the hinge loss and its gradient.
            Update Parameters: Update the weights and bias using the gradients.
            Stopping Criteria: Check for convergence based on the tolerance level
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
                
        # Gradient Descent Loop
        for i in range(self.max_iter):
            # Compute the margin
            margin = y * (np.dot(X, self.w) + self.b)
            
            # Compute the hinge loss and its gradient
            loss = np.maximum(0, 1 - margin)
            dw = np.zeros(n_features)
            db = 0.0
            
            # For each sample, update the gradient
            for j in range(n_samples):
                if loss[j] > 0:
                    dw -= y[j] * X[j]
                    db -= y[j]
            
            # Average the gradients
            dw = dw / n_samples + self.C * self.w
            db = db / n_samples
            
            # Update the weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Check for convergence
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

    def predict(self, X):
        """
        Predict class labels for samples in X.
        """
        return super().predict(X)
    
    def __sklearn_is_fitted__(self):
        return self.w is not None

class LinearSVR(BaseSVM):
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, epsilon=0.1):
        super().__init__(C, tol, max_iter, learning_rate)
        self.epsilon = epsilon
        
    def _fit(self, X, y):
        """
        Implement the fitting procedure for LinearSVR using the epsilon-insensitive loss.
        
        Algorithm:
            Initialize Parameters: Initialize the weight vector w and bias b.
            Set Hyperparameters: Define the learning rate and the number of iterations.
            Gradient Descent Loop: Iterate over the dataset to update the weights and bias using gradient descent.
            Compute Epsilon-Insensitive Loss: Calculate the epsilon-insensitive loss and its gradient.
            Update Parameters: Update the weights and bias using the gradients.
            Stopping Criteria: Check for convergence based on the tolerance level
        """
        # Initialize parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        epsilon = self.epsilon
                
        # Gradient Descent Loop
        for i in range(self.max_iter):
            # Compute the prediction
            prediction = np.dot(X, self.w) + self.b
            
            # Compute the epsilon-insensitive loss and its gradient
            loss = np.maximum(0, np.abs(y - prediction) - epsilon)
            dw = np.zeros(n_features)
            db = 0.0
            
            # For each sample, update the gradient
            for j in range(n_samples):
                if loss[j] > 0:
                    if y[j] > prediction[j] + epsilon:
                        dw -= X[j]
                        db -= 1
                    elif y[j] < prediction[j] - epsilon:
                        dw += X[j]
                        db += 1
            
            # Average the gradients
            dw = dw / n_samples + self.C * self.w
            db = db / n_samples
            
            # Update the weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Check for convergence
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break

    def predict(self, X):
        """
        Predict continuous values for samples in X.
        """
        return self.decision_function(X)

    def __sklearn_is_fitted__(self):
        return self.w is not None
