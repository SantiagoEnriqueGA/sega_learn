import numpy as np

class PCA:
    """
    Principal Component Analysis (PCA) implementation.
    """
    def __init__(self, n_components):
        """
        Principal Component Analysis (PCA) implementation.
        Uses the eigendecomposition of the covariance matrix to project the data onto a lower-dimensional space.
        
        Parameters:
        - n_components: number of principal components to keep
        """
        self.n_components = n_components
        self.components = None
        self.mean_ = None

    def fit(self, X):
        """
        Fit the PCA model to the data X.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
             The data to fit the model to.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        if self.n_components > X.shape[1]:
            raise ValueError("Number of components cannot be greater than the number of features.")
        
        # Mean centering
        self.mean_ = np.mean(X, axis=0)
        X = X - self.mean_

        # Covariance matrix
        cov = np.cov(X.T)

        # Eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # Sort eigenvectors by eigenvalues in descending order
        eigenvectors = eigenvectors[:, np.argsort(eigenvalues)[::-1]]

        # Select the top n_components eigenvectors
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components] / np.sum(eigenvalues)

    def transform(self, X):
        """
        Apply the dimensionality reduction on X.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
             The data to transform.
        
        Returns:
        - X_transformed: numpy array of shape (n_samples, n_components)
                         The data transformed into the principal component space.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError("Input data must have the same number of features as the data used to fit the model.")
        
        # Project data to the principal component space
        X = X - self.mean_
        return np.dot(X, self.components_)

    def fit_transform(self, X):
        """
        Fit the PCA model to the data X and apply the dimensionality reduction on X.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
             The data to fit the model to and transform.
        
        Returns:
        - X_transformed: numpy array of shape (n_samples, n_components)
                         The data transformed into the principal component space.
        """
        self.fit(X)
        return self.transform(X)
    
    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio_
    
    def get_explained_variance_ratio(self):
        return self.explained_variance_ratio_

    def get_components(self):
        return self.components_

    def inverse_transform(self, X_reduced):
        if not isinstance(X_reduced, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if X_reduced.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        return np.dot(X_reduced, self.components_.T) + self.mean_

class SVD:
    """
    Singular Value Decomposition (SVD) implementation.
    """
    def __init__(self, n_components):
        """
        Singular Value Decomposition (SVD) implementation.
        
        Parameters:
        - n_components: number of singular values and vectors to keep
        """
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, X):
        """
        Fit the SVD model to the data X.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
             The data to fit the model to.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        if self.n_components > min(X.shape):
            raise ValueError("Number of components cannot be greater than the minimum dimension of the input data.")
        
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.U = U[:, :self.n_components]
        self.S = S[:self.n_components]
        self.Vt = Vt[:self.n_components, :]

    def transform(self, X):
        """
        Apply the SVD transformation on X.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
             The data to transform.
        
        Returns:
        - X_transformed: numpy array of shape (n_samples, n_components)
                         The data transformed into the singular value space.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if X.ndim != 2:
            raise ValueError("Input data must be a 2D array.")
        
        return np.dot(X, self.Vt.T)

    def fit_transform(self, X):
        """
        Fit the SVD model to the data X and apply the SVD transformation on X.
        
        Parameters:
        - X: numpy array of shape (n_samples, n_features)
             The data to fit the model to and transform.
        
        Returns:
        - X_transformed: numpy array of shape (n_samples, n_components)
                         The data transformed into the singular value space.
        """
        self.fit(X)
        return self.transform(X)
    
    def get_singular_values(self):
        return self.S
    
    def get_singular_vectors(self):
        return self.U, self.Vt
