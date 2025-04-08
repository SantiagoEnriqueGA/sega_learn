# sega_learn/examples/auto/regressor.py
import os
import sys

# Adjust path to import from the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.auto import AutoRegressor
from sega_learn.utils import make_regression
from sega_learn.utils.metrics import Metrics

# Define metrics for potential custom use (though defaults are used below)
r_squared = Metrics.r_squared
root_mean_squared_error = Metrics.root_mean_squared_error
mean_absolute_error = Metrics.mean_absolute_error


def run_example(verbose=False):
    """Runs the example demonstrating AutoRegressor with and without tuning."""
    X, y = make_regression(
        n_samples=500, n_features=8, noise=15, random_state=42
    )  # Smaller dataset for faster example

    # --- 1. Run without Tuning (Baseline) ---
    print("\n--- Running AutoRegressor with Default Parameters ---")
    reg_default = AutoRegressor(all_kernels=False)  # Include RBF/Poly SVM kernels
    reg_default.fit(X, y, verbose=verbose)
    print("\nDefault Summary:")
    reg_default.summary()

    # --- 2. Run with Hyperparameter Tuning (Random Search) ---
    print("\n--- Running AutoRegressor with Hyperparameter Tuning (Random Search) ---")
    # Note: Tuning takes longer! Using fewer iterations/folds for the example.
    reg_tuned_random = AutoRegressor(
        all_kernels=False,
        tune_hyperparameters=True,
        tuning_method="random",  # Use random search
        tuning_iterations=5,  # Number of random combinations to try per model
        cv=2,  # Number of cross-validation folds
        tuning_metric="r2",  # Optimize for R-squared (maximize)
        # You could also use 'neg_mean_squared_error', 'rmse', 'mae', 'mape' (minimize)
    )
    reg_tuned_random.fit(
        X, y, verbose=verbose
    )  # verbose=True shows tuning progress if tqdm is installed
    print("\nTuned Summary (Random Search):")
    reg_tuned_random.summary()

    # --- Optional: Grid Search Example (can be slow) ---
    # print("\n--- Running AutoRegressor with Hyperparameter Tuning (Grid Search) ---")
    # reg_tuned_grid = AutoRegressor(
    #     all_kernels=True,
    #     tune_hyperparameters=True,
    #     tuning_method="grid",       # Use grid search
    #     cv=2,                       # Keep low for example speed
    #     tuning_metric="neg_mean_squared_error" # Optimize for lowest MSE
    # )
    # reg_tuned_grid.fit(X, y, verbose=verbose)
    # print("\nTuned Summary (Grid Search):")
    # reg_tuned_grid.summary()

    # --- 3. Prediction and Evaluation using the Tuned Model ---
    print("\n--- Predictions and Evaluations using Tuned (Random) Model ---")
    # Use the instance that performed tuning for subsequent steps
    reg = reg_tuned_random

    # Predict using all fitted (and potentially tuned) models
    # Using a small subset of data for prediction display
    X_pred_subset = X[:5]
    predictions = reg.predict(X_pred_subset)
    print("\nPredictions for first 5 samples (Tuned Models):")
    for model, pred in predictions.items():
        # Ensure pred is displayed nicely, handle potential errors during prediction
        if isinstance(pred, str) and pred.startswith("Prediction Error"):
            print(f"\t{model}: {pred}")
        else:
            print(f"\t{model}: {pred.round(2)}")

    # Predict using a specific model (e.g., the potentially tuned RandomForestRegressor)
    try:
        rfr_pred = reg.predict(X_pred_subset, model="RandomForestRegressor")
        print(f"\nRandomForestRegressor Predictions (Tuned): {rfr_pred.round(2)}")
    except ValueError as e:
        print(f"\nCould not get specific prediction for RandomForestRegressor: {e}")
    except Exception as e:
        print(
            f"\nAn unexpected error occurred predicting with RandomForestRegressor: {e}"
        )

    # Evaluate all fitted models on the original target data 'y'
    print("\n--- Evaluation on Training Data (Tuned Models) ---")
    results = reg.evaluate(y)  # Evaluate against the original training labels
    print("\nAll Evaluation Results (Tuned):")
    for model, result in results.items():
        # Format metrics nicely
        formatted_result = {
            k: f"{v:.4f}" if isinstance(v, int | float) else v
            for k, v in result.items()
        }
        print(f"\t{model}: {formatted_result}")

    # Evaluate a specific model
    try:
        results_rfr = reg.evaluate(y, model="RandomForestRegressor")
        formatted_rfr_result = {
            k: f"{v:.4f}" if isinstance(v, int | float) else v
            for k, v in results_rfr["RandomForestRegressor"].items()
        }
        print(f"\nRandomForestRegressor Results (Tuned): {formatted_rfr_result}")
    except ValueError as e:
        print(f"\nCould not get specific evaluation for RandomForestRegressor: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred evaluating RandomForestRegressor: {e}")

    # --- 4. Accessing the Best Tuned Model ---
    print("\n--- Accessing Best Model ---")
    # Find the best model name from the summary results (assuming first is best after sort)
    if reg.results:
        # Find the best result that doesn't have an Error key
        best_result = next((r for r in reg.results if "Error" not in r), None)
        if best_result:
            best_model_name = best_result["Model"]
            print(
                f"Best model based on {reg.tuning_metric} (from summary): {best_model_name}"
            )
            try:
                # Get the actual fitted model instance
                best_model_instance = reg.get_model(best_model_name)
                print(f"Instance of best model retrieved: {type(best_model_instance)}")
                # You could now use 'best_model_instance' directly for more analysis or saving
            except ValueError as e:
                print(f"Could not retrieve best model instance: {e}")
        else:
            print("Could not determine the best model from results.")
    else:
        print("No results available to determine the best model.")


if __name__ == "__main__":
    run_example(verbose=True)  # Set verbose=True to see fitting/tuning steps
