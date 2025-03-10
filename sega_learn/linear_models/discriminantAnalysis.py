# Importing the required libraries
from random import shuffle
import numpy as np
from math import log, floor, ceil
from scipy import linalg

class LinearDiscriminantAnalysis(object):
    """
    Implements Linear Discriminant Analysis.
    A classifier with a linear decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.
    
    Parameters:
    - solver : {'svd', 'lsqr', 'eigen'}, default='svd'
        Solver to use for the LDA. 
            'svd' is the default and recommended solver. 
            'lsqr' is a faster alternative that can be used when the number of features is large. 
            'eigen' is an alternative solver that can be used when the number of features is small.
    - priors : array-like, shape (n_classes,), default=None
        Prior probabilities of the classes. If None, the priors are uniform.
    """
    def __init__(self, solver='svd', priors=None):
        """
        Initialize the Linear Discriminant Analysis model with the specified solver and prior probabilities.
        
        Parameters:
        - solver : {'svd', 'lsqr', 'eigen'}, default='svd'
            Solver to use for the LDA. 
                'svd' is the default and recommended solver. 
                'lsqr' is a faster alternative that can be used when the number of features is large. 
                'eigen' is an alternative solver that can be used when the number of features is small.
        - priors : array-like, shape (n_classes,), default=None
            Prior probabilities of the classes. If None, the priors are uniform.
        """
        self.solver = solver
        self.priors = priors
        
    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        This method computes the mean and covariance of each class, and the prior probabilities of each class.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
        - y : array-like, shape (n_samples,): Target values, i.e., class labels.
        """
        self.classes_ = np.unique(y)                # Unique classes, i.e., distinct class labels
        self.means_ = {}                            # Mean of each feature per class
        self.covariance_ = np.cov(X, rowvar=False)  # Covariance matrix of all features
        if self.priors is None:                     # Prior probabilities of each class
            self.priors_ = {}
        else:
            self.priors_ = self.priors
            
        # Compute mean and prior for each class, and covariance
        for cls in self.classes_:
            X_cls = X[y == cls]                                 # Data points corresponding to class cls
            self.means_[cls] = np.mean(X_cls, axis=0)           # Mean of each feature per class
            self.priors_[cls] = X_cls.shape[0] / X.shape[0]     # Prior probability of class cls
        
        if self.solver == 'svd':
            self._fit_svd(X, y)
        elif self.solver == 'lsqr':
            self._fit_lsqr(X, y)
        elif self.solver == 'eigen':
            self._fit_eigen(X, y)
        else:
            raise ValueError(f"Solver '{self.solver}' is not yet supported.")
    
    def _fit_svd(self, X, y):
        """
        Fit the model using Singular Value Decomposition.
        Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three other matrices. 
        In the context of LDA, SVD is used to find the linear combinations of features that best separate the classes.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
        - y : array-like, shape (n_samples,): Target values, i.e., class labels.
        """
        X_centered = X - np.mean(X, axis=0)                         # Center the data
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)   # Perform SVD, U and Vt are the left and right singular vectors, S is the singular values
        rank = np.sum(S > 1e-10)                                    # Compute the rank of the matrix, i.e., the number of singular values greater than 1e-10
        
        # Select only the top components
        U = U[:, :rank]
        S = S[:rank]
        Vt = Vt[:rank, :]
        
        self.scalings_ = Vt.T / S                                                           # Compute the transformation matrix, i.e., the scalings
        self.means_ = {cls: mean @ self.scalings_ for cls, mean in self.means_.items()}     # Transform the means, i.e., the mean of each feature per class
        self.covariance_ = np.diag(1 / S**2)                                                # Transform the covariance matrix, i.e., the inverse of the singular values squared
        
    def _fit_lsqr(self, X, y):
        """
        Fit the model using LSQR (Least Squares).
        LSQR (Least Squares) is a method for solving linear equations. 
        In the context of LDA, LSQR is used to find the linear combinations of features that best separate the classes by solving a least squares problem.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
        - y : array-like, shape (n_samples,): Target values, i.e., class labels.
        """
        # Create matrix for class-specific means
        mean_matrix = np.vstack([self.means_[cls] for cls in self.classes_])

        # Solve least squares problem
        X_centered = X - np.mean(X, axis=0)
        Y = np.zeros((X.shape[0], len(self.classes_)))
        for i, cls in enumerate(self.classes_):
            Y[:, i] = (y == cls).astype(float)

        # Solve the normal equations using least squares
        coef, residuals, rank, singular_values = linalg.lstsq(X_centered, Y)

        # Compute scalings
        self.scalings_ = coef

        # Transform class means and store them back in the dictionary
        self.means_ = {cls: (self.means_[cls] @ self.scalings_) for cls in self.classes_}
        
        # Update the covariance matrix to match the transformed data
        self.covariance_ = np.cov(X_centered @ self.scalings_, rowvar=False)

    def _fit_eigen(self, X, y):
        """
        Fit the model using eigenvalue decomposition.
        Eigenvalue decomposition is a method for decomposing a matrix into its eigenvalues and eigenvectors.
        In the context of LDA, eigenvalue decomposition is used to find the linear combinations of features that best separate the classes by solving a generalized eigenvalue problem.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
        - y : array-like, shape (n_samples,): Target values, i.e., class labels.
        """
        # Center the data by subtracting the global mean
        X_centered = X - np.mean(X, axis=0)

        # Compute the total scatter matrix (within-class + between-class)
        # We'll compute this by first computing the within-class scatter matrix
        Sw = np.zeros((X.shape[1], X.shape[1]))
        Sb = np.zeros((X.shape[1], X.shape[1]))

        for cls in self.classes_:
            X_cls = X[y == cls]
            X_cls_centered = X_cls - self.means_[cls]
            
            # Within-class scatter matrix
            Sw += X_cls_centered.T @ X_cls_centered
            
            # Between-class scatter matrix
            cls_mean_centered = self.means_[cls] - np.mean(X, axis=0)
            Sb += len(X_cls) * (cls_mean_centered[:, np.newaxis] @ cls_mean_centered[np.newaxis, :])

        # Solve the generalized eigenvalue problem: Sb @ w = λ * Sw @ w
        # This requires inverting the within-class scatter matrix
        eigenvalues, eigenvectors = linalg.eigh(np.linalg.inv(Sw) @ Sb)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select top eigenvectors
        n_components = len(self.classes_)
        self.scalings_ = eigenvectors[:, :n_components]

        # Transform class means
        self.means_ = {cls: (self.means_[cls] @ self.scalings_) for cls in self.classes_}

        # Transform the covariance matrix
        # Use the inverse of the eigenvalues as the diagonal of the transformed covariance
        self.covariance_ = np.diag(1.0 / (eigenvalues[:n_components] + 1e-11))
    
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.
        
        Returns:
        - array, shape (n_samples,): Predicted class labels for the input samples.
        """
        scores = self.decision_function(X)              # Compute the decision function
        return self.classes_[np.argmax(scores, axis=1)] # Return the class with the highest score
    
    def decision_function(self, X):
        """
        Apply decision function to an array of samples. 
        The decision function is the log-likelihood of each class.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.
        
        Returns:
        - array, shape (n_samples, n_classes): Log-likelihood of each class for the input samples.
        """
        scores = []
        # Compute log-likelihood of each class
        for cls in self.classes_:
            mean = self.means_[cls]     # Mean of each feature per class
            prior = self.priors_[cls]   # Prior probability of class cls

            # Score is the log-likelihood of each class
            score = X @ np.linalg.inv(self.covariance_) @ mean.T - 0.5 * mean @ np.linalg.inv(self.covariance_) @ mean.T + log(prior)
            
            scores.append(score)        # Append the score to the list of scores
        return np.array(scores).T       # Return the scores as a numpy array, with each row corresponding to a sample and each column corresponding to a class

class QuadraticDiscriminantAnalysis(object):
    """
    Implements Quadratic Discriminant Analysis.
    The quadratic term allows for more flexibility in modeling the class conditional
    A classifier with a quadratic decision boundary, generated by fitting class conditional densities to the data and using Bayes' rule.
    
    Parameters:
    - priors : array-like, shape (n_classes,), default=None
        Prior probabilities of the classes. If None, the priors are uniform.
    - reg_param : float, default=0.0
        Regularization parameter. If greater than 0, the covariance matrices are regularized by adding a scaled identity matrix to them.
    """
    def __init__(self, priors=None, reg_param=0.0):
        """
        Initialize the Quadratic Discriminant Analysis model with the specified prior probabilities and regularization parameter.
        
        Parameters:
        - priors : array-like, shape (n_classes,), default=None
            Prior probabilities of the classes. If None, the priors are uniform.
        - reg_param : float, default=0.0
        """
        self.priors = priors
        
        assert reg_param >= 0.0, "Regularization parameter must be non-negative."
        self.reg_param = reg_param
        
    def fit(self, X, y):
        """
        Fit the model according to the given training data. Uses the means and covariance matrices of each class.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Training data, where n_samples is the number of samples and n_features is the number of features.
        - y : array-like, shape (n_samples,): Target values, i.e., class labels.
        """
        self.classes_ = np.unique(y)                # Unique classes, i.e., distinct class labels
        self.means_ = {}                            # Mean of each feature per class
        self.covariances_ = {}                      # Covariance matrix of each class
        if self.priors is None:                     # Prior probabilities of each class
            self.priors_ = {}
        else:
            self.priors_ = self.priors
            
        # Compute mean and prior for each class, and covariance
        for cls in self.classes_:
            X_cls = X[y == cls]                                 # Data points corresponding to class cls
            self.means_[cls] = np.mean(X_cls, axis=0)           # Mean of each feature per class
            self.priors_[cls] = X_cls.shape[0] / X.shape[0]     # Prior probability of class cls
            
            cov = np.cov(X_cls, rowvar=False)                   # Covariance matrix of all features
            
            if self.reg_param > 0.0:
                self.covariances_[cls] = cov + np.eye(cov.shape[0]) * self.reg_param  # Add regularization term to diagonal
            else:
                self.covariances_[cls] = cov + np.eye(cov.shape[0]) * 1e-6  # Add small value to diagonal
    
    def predict(self, X):
        """
        Perform classification on an array of test vectors X.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.
        
        Returns:
        - array, shape (n_samples,): Predicted class labels for the input samples.
        """
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]
    
    def decision_function(self, X):
        """
        Apply decision function to an array of samples.
        The decision function is the log-likelihood of each class.
        
        Parameters:
        - X : array-like, shape (n_samples, n_features): Test data, where n_samples is the number of samples and n_features is the number of features.
        
        Returns:
        - array, shape (n_samples, n_classes): Log-likelihood of each class for the input samples.
        """
        scores = []
        # Compute log-likelihood of each class
        for cls in self.classes_:
            mean = self.means_[cls]     # Mean of each feature per class
            cov = self.covariances_[cls] # Covariance matrix of each class
            prior = self.priors_[cls]   # Prior probability of class cls
            
            # Compute the inverse of the covariance matrix and the log-determinant of the covariance matrix
            inv_cov = np.linalg.inv(cov)
            log_det_cov = np.log(np.linalg.det(cov))
            
            # Compute the log-likelihood
            score = -0.5 * np.sum((X - mean) @ inv_cov * (X - mean), axis=1)    # Quadratic term
            score -= 0.5 * log_det_cov                                          # Log-determinant term
            score += log(prior)                                                 # Prior term
            
            scores.append(score)    # Append the score to the list of scores
        return np.array(scores).T   # Return the scores as a numpy array, with each row corresponding to a sample and each column corresponding to a class


def make_sample_data(n_samples, n_features, cov_class_1, cov_class_2, shift=[1,1], seed=0):
    """
    Make data for testing, for testing LDA and QDA. 
    Data points for class 1 are generated by multiplying a random matrix with the 
        covariance matrix of class 1, and data points for class 2 are generated by multiplying 
        a random matrix with the covariance matrix of class 2 and adding [1, 1].
    """
    rng = np.random.RandomState(seed)
    X = np.concatenate(
        [
            # Data points for class 1, generated by multiplying a random matrix with the covariance matrix of class 1
            rng.randn(n_samples, n_features) @ cov_class_1,
            # Data points for class 2, generated by multiplying a random matrix with the covariance matrix of class 2 and adding [1, 1]
            rng.randn(n_samples, n_features) @ cov_class_2 + np.array(shift),
        ]
    )
    y = np.concatenate([np.zeros(n_samples), np.ones(n_samples)])
    return X, y

