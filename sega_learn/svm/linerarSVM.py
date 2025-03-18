from .baseSVM import BaseSVM
import numpy as np

class LinearSVC(BaseSVM):
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01):
        super().__init__(C, tol, max_iter, learning_rate)
        
    def _fit(self, X, y):
        """
        Implement the fitting procedure for LinearSVC using gradient descent.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Training vectors.
            y (array-like of shape (n_samples,)): Target labels in {-1, 1}.
            
        Returns:
            self (LinearSVC): The fitted instance.

        Algorithm:
            Initialize Parameters: Initialize the weight vector w and bias b.
            Set Hyperparameters: Define the learning rate and the number of iterations.
            Gradient Descent Loop: Iterate over the dataset to update the weights and bias using gradient descent.
            Compute Hinge Loss: Calculate the hinge loss and its gradient.
            Update Parameters: Update the weights and bias using the gradients.
            Stopping Criteria: Check for convergence based on the tolerance level
        """
        if self.kernel != 'linear':
            raise ValueError("LinearSVC only supports linear kernel")

        # Initialize parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Convert y to {-1, 1} if not already
        y = np.where(y <= 0, -1, 1)
                
        # Gradient Descent Loop
        for _ in range(self.max_iter):
            # Compute the margin
            margin = y * (np.dot(X, self.w) + self.b)
            
            # Compute the hinge loss and its gradient
            violated_indices = margin < 1
            
            # Initialize gradients
            dw = self.C * self.w  # L2 regularization gradient
            db = 0.0
            
            # Update gradients based on violated constraints
            if np.any(violated_indices):
                X_violated = X[violated_indices]
                y_violated = y[violated_indices]
                dw -= np.sum(X_violated * y_violated[:, np.newaxis], axis=0) / n_samples
                db -= np.sum(y_violated) / n_samples
            
            # Update the weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Calculate current loss for convergence check
            loss = self.C * 0.5 * np.dot(self.w, self.w) + np.sum(np.maximum(0, 1 - margin)) / n_samples
            
            # Check for convergence based on gradient norm
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break
                
        return self
   
    def _predict_binary(self, X):
        """
        Predict class labels for binary classification.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Input samples.
        
        Returns:
            y_pred (array of shape (n_samples,)): Predicted class labels {-1, 1}.
        """
        return np.sign(self.decision_function(X))
    
    def _predict_multiclass(self, X):
        """
        Predict class labels for multi-class classification using one-vs-rest strategy.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Input samples.
        
        Returns:
            predicted_labels (array of shape (n_samples,)): Predicted class labels.    
        """
        decision_values = np.array([model.decision_function(X) for model in self.models_]).T
        return self.classes_[np.argmax(decision_values, axis=1)]
        
    
    def decision_function(self, X):
        """
        Compute raw decision function values before thresholding.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Input samples.
            
        Returns:
            scores (array of shape (n_samples,)): Decision function values.
        """
        return super().decision_function(X)
    
    def _score_binary(self, X, y):
        """
        Compute the mean accuracy of predictions for binary classification.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Test samples.
            y (array-like of shape (n_samples,)): True labels.
            
        Returns:
            score (float): Mean accuracy of predictions.
        """
        y_true = np.where(y <= 0, -1, 1)
        y_pred = self.predict(X)
        return np.mean(y_true == y_pred)
    
    def _score_multiclass(self, X, y):
        """
        Compute the mean accuracy of predictions for multi-class classification.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Test samples.
            y (array-like of shape (n_samples,)): True labels.
            
        Returns:
            score (float): Mean accuracy of predictions.
        """
        y_pred = self.predict(X)
        return np.mean(y == y_pred)


class LinearSVR(BaseSVM):
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000, learning_rate=0.01, epsilon=0.1):
        super().__init__(C, tol, max_iter, learning_rate, regression=True)
        self.epsilon = epsilon
        
    def _fit(self, X, y):
        """
        Implement the fitting procedure for LinearSVR using the epsilon-insensitive loss.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Training vectors.
            y (array-like of shape (n_samples,)): Target values.
            
        Returns:
            self (LinearSVR): The fitted instance.
        
        Algorithm:
            Initialize Parameters: Initialize the weight vector w and bias b.
            Set Hyperparameters: Define the learning rate and the number of iterations.
            Gradient Descent Loop: Iterate over the dataset to update the weights and bias using gradient descent.
            Compute Epsilon-Insensitive Loss: Calculate the epsilon-insensitive loss and its gradient.
            Update Parameters: Update the weights and bias using the gradients.
            Stopping Criteria: Check for convergence based on the tolerance level
        """
        if self.kernel != 'linear':
            raise ValueError("LinearSVR only supports linear kernel")
        
        # Initialize parameters
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Store for prediction
        self.X_train = X
        self.y_train = y
                
        # Gradient Descent Loop
        for i in range(self.max_iter):
            # Compute the prediction
            prediction = np.dot(X, self.w) + self.b
            
            # Compute the epsilon-insensitive loss and its gradient
            errors = y - prediction
            abs_errors = np.abs(errors)
            
            # Initialize gradients with regularization term
            dw = self.C * self.w
            db = 0.0
            
            # Samples outside the epsilon tube (positive errors)
            pos_idx = errors > self.epsilon
            if np.any(pos_idx):
                dw -= np.sum(X[pos_idx], axis=0) / n_samples
                db -= np.sum(np.ones(np.sum(pos_idx))) / n_samples
            
            # Samples outside the epsilon tube (negative errors)
            neg_idx = errors < -self.epsilon
            if np.any(neg_idx):
                dw += np.sum(X[neg_idx], axis=0) / n_samples
                db += np.sum(np.ones(np.sum(neg_idx))) / n_samples
            
            # Update the weights and bias
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            
            # Calculate current loss for convergence check
            epsilon_loss = np.sum(np.maximum(0, abs_errors - self.epsilon)) / n_samples
            reg_loss = self.C * 0.5 * np.dot(self.w, self.w)
            total_loss = epsilon_loss + reg_loss
            
            # Check for convergence based on gradient norm
            if np.linalg.norm(dw) < self.tol and abs(db) < self.tol:
                break
                
        return self

    def predict(self, X):
        """
        Predict continuous target values for input samples.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Input samples.
            
        Returns:
            y_pred (array of shape (n_samples,)): Predicted values.
        """
        return self.decision_function(X)
    
    def decision_function(self, X):
        """
        Compute raw decision function values.
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Input samples.
            
        Returns:
            scores (array of shape (n_samples,)): Predicted values.
        """
        return super().decision_function(X)
    
    def score(self, X, y):
        """
        Compute the coefficient of determination (R² score).
        
        Parameters:
            X (array-like of shape (n_samples, n_features)): Test samples.
            y (array-like of shape (n_samples,)): True target values.
            
        Returns:
            score (float): R² score of predictions.
        """
        y_pred = self.predict(X)
        u = ((y - y_pred) ** 2).sum()
        v = ((y - y.mean()) ** 2).sum()
        return 1 - u/v if v > 0 else 0

