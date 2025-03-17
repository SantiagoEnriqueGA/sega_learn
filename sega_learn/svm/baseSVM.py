import numpy as np

class BaseSVM:
    def __init__(self, C=1.0, tol=1e-4, max_iter=1000):
        self.C = C            # Regularization parameter
        self.tol = tol        # Tolerance for stopping criteria
        self.max_iter = max_iter  # Maximum number of iterations
        self.w = None         # Weight vector
        self.b = 0.0          # Bias term

    def fit(self, X, y):
        """
        Validate input data and initiate the fitting procedure.
        """
        X = np.array(X)
        y = np.array(y)
        # Perform input checks here if needed.
        
        # Let the subclass handle the actual optimization.
        self._fit(X, y)
        return self

    def _fit(self, X, y):
        """
        The actual fitting procedure to be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this!")

    def decision_function(self, X):
        """
        Compute the decision function: f(x) = <w, x> + b
        """
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        """
        For classification tasks, return the sign of the decision function.
        This method can be overridden by subclasses if needed.
        """
        scores = self.decision_function(X)
        return np.where(scores >= 0, 1, -1)

