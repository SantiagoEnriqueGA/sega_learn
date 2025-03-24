import numpy as np
cimport numpy as cnp  # Import Cython's NumPy interface

# Declare the types for better performance
def fit_ols(cnp.ndarray[cnp.float64_t, ndim=2] X, cnp.ndarray[cnp.float64_t, ndim=1] y, bint fit_intercept=True):
    """
    Fit the Ordinary Least Squares model using Cython.
    """
    cdef int n_samples = X.shape[0]
    cdef int n_features = X.shape[1]
    cdef cnp.ndarray[cnp.float64_t, ndim=2] X_with_intercept
    cdef cnp.ndarray[cnp.float64_t, ndim=1] coef

    if fit_intercept:
        # Add a column of ones for the intercept
        X_with_intercept = np.hstack([np.ones((n_samples, 1)), X])
    else:
        X_with_intercept = X

    # Compute coefficients using the normal equation: (X^T * X)^-1 * X^T * y
    coef = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

    return coef


