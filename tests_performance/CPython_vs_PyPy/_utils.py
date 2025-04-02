import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_print():
    """
    A context manager that suppresses all print statements within its block.
    This function redirects the standard output to os.devnull, silencing any print statements executed within its context.
    Once the context is exited, the standard output is restored to its original state.
    Usage:
        with suppress_print():
            # Any print statements here will be suppressed
    Yields:
        None
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def synthetic_data_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42):
    """
    Generate synthetic regression data for testing.

    Parameters:
    - n_samples: int, number of samples to generate
    - n_features: int, number of features (independent variables)
    - noise: float, standard deviation of the Gaussian noise added to the output
    - random_state: int or None, seed used by the random number generator

    Returns:
    - X: array-like, shape (n_samples, n_features), generated features
    - y: array-like, shape (n_samples,), generated target values
    """
    from sega_learn.utils import make_regression

    return make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state,
    )


def strip_file_path(file_path):
    """
    Strips the file path and returns the file name.
    """
    return os.path.basename(file_path)
