import os
import sys
import time

from utils import suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.utils import *
from sega_learn.linear_models import make_sample_data
from utils import synthetic_data_regression, suppress_print

from sklearn.preprocessing import PolynomialFeatures

NUM_SAMPLES = 1_000_000
NUM_RUNS = 5

def test_polynomial_transform():
    # Generate synthetic data
    X, _ = synthetic_data_regression(n_samples=NUM_SAMPLES, n_features=5, noise=0.1, random_state=42)

    def sega_poly_transform():
        start_time = time.time()
        model = PolynomialTransform(degree=2)
        X_poly = model.fit_transform(X)
        end_time = time.time()
        return end_time - start_time

    def sklearn_poly_transform():
        start_time = time.time()
        model = PolynomialFeatures(degree=2)
        X_poly = model.fit_transform(X)
        end_time = time.time()
        return end_time - start_time

    sklearn_times = [sklearn_poly_transform() for _ in range(NUM_RUNS)]
    sega_times = [sega_poly_transform() for _ in range(NUM_RUNS)]

    avg_sklearn_time = sum(sklearn_times) / NUM_RUNS
    avg_sega_time = sum(sega_times) / NUM_RUNS

    print(f"\nPolynomial Transform Performance comparison on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
    print(f"\tsklearn Polynomial Features: {avg_sklearn_time:7.4f} seconds")
    print(f"\tsega Polynomial Transform: {avg_sega_time:10.4f} seconds")

    if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
    if avg_sega_time == 0: avg_sega_time = 1e-10
    if avg_sklearn_time == avg_sega_time:
        print("\tSpeed difference is negligible")
    elif avg_sklearn_time > avg_sega_time:
        print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
    else:
        print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")

def main():
    test_polynomial_transform()

if __name__ == '__main__':
    main()
