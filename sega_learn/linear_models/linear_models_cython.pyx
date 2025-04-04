# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport fabs

#--------------------------------------------------------------------
# Ordinary Least Squares using the Normal Equation
#--------------------------------------------------------------------
def fit_ols(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=1] y, bint fit_intercept):
    """
    Compute OLS coefficients.

    If fit_intercept is True, a column of ones is prepended to X.
    Returns a 1D array of coefficients (with the first element being the intercept, if applicable).
    """
    # Explicit cast to int
    cdef int n_samples = <int>X.shape[0]
    cdef int n_features = <int>X.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] X_design

    if fit_intercept:
        X_design = np.empty((n_samples, n_features + 1), dtype=np.double)
        X_design[:, 0] = 1.0
        X_design[:, 1:] = X
    else:
        X_design = X

    # Compute X_design^T * X_design and X_design^T * y
    cdef np.ndarray[np.double_t, ndim=2] XtX = np.dot(X_design.T, X_design)
    cdef np.ndarray[np.double_t, ndim=1] Xty = np.dot(X_design.T, y)

    # Solve the linear system XtX * coef = Xty
    cdef np.ndarray[np.double_t, ndim=1] coef = np.linalg.solve(XtX, Xty)
    return coef

#--------------------------------------------------------------------
# Ridge Regression using Coordinate Descent
#--------------------------------------------------------------------
def fit_ridge(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=1] y,
              double alpha, bint fit_intercept, int max_iter=10000, double tol=1e-4):
    """
    Compute Ridge regression coefficients using coordinate descent.

    Args:
    - X: Feature matrix.
    - y: Target vector.
    - alpha: Regularization strength.
    - fit_intercept: Whether to include an intercept.
    - max_iter: Maximum iterations.
    - tol: Convergence tolerance.

    Returns a 1D array of coefficients (with intercept in position 0 if fit_intercept is True).
    """
    # Explicit cast to int
    cdef int n_samples = <int>X.shape[0]
    cdef int n_features = <int>X.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] X_design

    if fit_intercept:
        X_design = np.empty((n_samples, n_features + 1), dtype=np.double)
        X_design[:, 0] = 1.0
        X_design[:, 1:] = X
        n_features += 1
    else:
        X_design = X

    cdef np.ndarray[np.double_t, ndim=1] coef = np.zeros(n_features, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] coef_old
    cdef np.ndarray[np.double_t, ndim=1] residual  # declare once outside the loops
    cdef int i, j
    cdef double rho, denom, sum_diff

    for i in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            residual = y - np.dot(X_design, coef)
            rho = np.dot(X_design[:, j], residual)
            if fit_intercept and j == 0:
                denom = np.dot(X_design[:, j], X_design[:, j])
                if denom != 0:
                    coef[j] = rho / denom
                else:
                    coef[j] = 0.0
            else:
                denom = np.dot(X_design[:, j], X_design[:, j]) + alpha
                if denom != 0:
                    coef[j] = rho / denom
                else:
                    coef[j] = 0.0
        sum_diff = np.sum(np.abs(coef - coef_old))
        if sum_diff < tol:
            break
    return coef

#--------------------------------------------------------------------
# Lasso Regression using Coordinate Descent
#--------------------------------------------------------------------
def fit_lasso(np.ndarray[np.double_t, ndim=2] X, np.ndarray[np.double_t, ndim=1] y,
              double alpha, bint fit_intercept, int max_iter=10000, double tol=1e-4):
    """
    Compute Lasso regression coefficients using coordinate descent.

    Args:
    - X: Feature matrix.
    - y: Target vector.
    - alpha: Regularization strength.
    - fit_intercept: Whether to include an intercept.
    - max_iter: Maximum iterations.
    - tol: Convergence tolerance.

    Returns a 1D array of coefficients (with intercept in position 0 if fit_intercept is True).
    """
    # Explicit cast to int
    cdef int n_samples = <int>X.shape[0]
    cdef int n_features = <int>X.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] X_design

    if fit_intercept:
        X_design = np.empty((n_samples, n_features + 1), dtype=np.double)
        X_design[:, 0] = 1.0
        X_design[:, 1:] = X
        n_features += 1
    else:
        X_design = X

    cdef np.ndarray[np.double_t, ndim=1] coef = np.zeros(n_features, dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=1] coef_old
    cdef np.ndarray[np.double_t, ndim=1] residual  # declare once outside the loops
    cdef int i, j
    cdef double rho, denom, sum_diff, updated

    for i in range(max_iter):
        coef_old = coef.copy()
        for j in range(n_features):
            residual = y - np.dot(X_design, coef)
            rho = np.dot(X_design[:, j], residual)
            if fit_intercept and j == 0:
                # Update intercept as simple average
                if n_samples != 0:
                    coef[j] = rho / n_samples
                else:
                    coef[j] = 0.0
            else:
                denom = np.dot(X_design[:, j], X_design[:, j])
                if denom == 0:
                    updated = 0.0
                else:
                    updated = np.sign(rho) * max(0.0, fabs(rho) - alpha) / denom
                coef[j] = updated
        sum_diff = np.sum(np.abs(coef - coef_old))
        if sum_diff < tol:
            break
    return coef
