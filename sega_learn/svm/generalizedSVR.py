from .baseSVM import BaseSVM
import numpy as np

class GeneralizedSVR(BaseSVM):
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, epsilon=0.1,
                 kernel='linear', degree=3, gamma='scale', coef0=0.0):
        super().__init__(C, tol, max_iter, learning_rate, kernel, degree, gamma, coef0, regression=True)
        self.epsilon = epsilon

    def _fit(self, X, y):
        """
        Fit the SVR model using gradient descent with support for multiple kernels.
        """
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features) if self.kernel == 'linear' else None
        self.b = 0.0
    
        # Initialize support vector attributes for non-linear kernels
        if self.kernel != 'linear':
            self.support_vector_alphas_ = np.zeros(n_samples)
            self.support_vector_labels_ = y
    
        # Store for prediction
        self.X_train = X
        self.y_train = y
    
        for _ in range(self.max_iter):
            # Compute predictions
            if self.kernel == 'linear':
                prediction = np.dot(X, self.w) + self.b
            else:
                K = self._compute_kernel(X, self.X_train)
                prediction = np.dot(K, self.support_vector_alphas_ * self.support_vector_labels_) + self.b
    
            # Compute epsilon-insensitive loss and gradients
            errors = y - prediction
            abs_errors = np.abs(errors)
    
            if self.kernel == 'linear':
                dw = self.C * self.w
                db = 0.0
    
                pos_idx = errors > self.epsilon
                neg_idx = errors < -self.epsilon
    
                if np.any(pos_idx):
                    dw -= np.dot(X[pos_idx].T, np.ones(np.sum(pos_idx))) / n_samples
                    db -= np.sum(np.ones(np.sum(pos_idx))) / n_samples
                if np.any(neg_idx):
                    dw += np.dot(X[neg_idx].T, np.ones(np.sum(neg_idx))) / n_samples
                    db += np.sum(np.ones(np.sum(neg_idx))) / n_samples
    
                # Update weights and bias
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
            else:
                # Gradient update for non-linear kernels
                pos_idx = errors > self.epsilon
                neg_idx = errors < -self.epsilon
    
                if np.any(pos_idx):
                    self.support_vector_alphas_[pos_idx] += self.learning_rate * (errors[pos_idx] - self.C * self.support_vector_alphas_[pos_idx])
                if np.any(neg_idx):
                    self.support_vector_alphas_[neg_idx] -= self.learning_rate * (errors[neg_idx] + self.C * self.support_vector_alphas_[neg_idx])
    
                # Update bias term
                db = -np.sum(errors[(pos_idx | neg_idx)]) / (n_samples + 1e-8)  # Add small value to avoid division by zero
                self.b -= self.learning_rate * db
    
            # Check for convergence
            if self.kernel == 'linear' and np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break
            elif self.kernel != 'linear' and abs(db) < self.tol:
                break
            
            if np.any(np.isnan(self.support_vector_alphas_)) or np.isnan(self.b):
                raise ValueError("Numerical instability detected during training. Consider normalizing the data or reducing the learning rate.")
    
        return self

    def predict(self, X):
        """
        Predict continuous target values for input samples.
        """
        return self.decision_function(X)

    def decision_function(self, X):
        """
        Compute raw decision function values.
        """
        if self.kernel == 'linear':
            return np.dot(X, self.w) + self.b
        else:
            K = self._compute_kernel(X, self.X_train)
            return np.dot(K, self.support_vector_alphas_ * self.support_vector_labels_) + self.b

    def score(self, X, y):
        """
        Compute the coefficient of determination (R² score).
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u / v if v > 0 else 0
