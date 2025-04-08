import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.auto import AutoClassifier
from sega_learn.utils import make_classification
from sega_learn.utils.metrics import Metrics

accuracy = Metrics.accuracy
precision = Metrics.precision
recall = Metrics.recall
f1 = Metrics.f1_score


def run_example(verbose=False):
    """Runs the example."""
    X, y = make_classification(n_samples=1_000, n_features=5, random_state=42)

    # --- Original ---
    # reg = AutoClassifier()
    # reg = AutoClassifier(all_kernels=True)

    # --- With Tuning Enabled ---
    print("\n--- Running AutoClassifier with Default Parameters ---")
    reg_default = AutoClassifier(all_kernels=False)
    reg_default.fit(X, y, verbose=verbose)
    print("\nDefault Summary:")
    reg_default.summary()

    print("\n--- Running AutoClassifier with Hyperparameter Tuning (Random Search) ---")
    # Note: Tuning takes longer! Use fewer iterations/folds for quick tests.
    reg_tuned_random = AutoClassifier(
        all_kernels=False,
        tune_hyperparameters=True,
        tuning_method="random",
        tuning_iterations=5,  # Keep low for example speed
        cv=2,  # Keep low for example speed
        tuning_metric="f1",  # Optimize for F1-score
    )
    reg_tuned_random.fit(X, y, verbose=verbose)
    print("\nTuned Summary (Random):")
    reg_tuned_random.summary()

    # --- Optional: Grid Search Example ---
    # print("\n--- Running AutoClassifier with Hyperparameter Tuning (Grid Search) ---")
    # reg_tuned_grid = AutoClassifier(
    #     all_kernels=True,
    #     tune_hyperparameters=True,
    #     tuning_method="grid",
    #     cv=2, # Keep low for example speed
    #     tuning_metric="accuracy" # Optimize for accuracy
    # )
    # reg_tuned_grid.fit(X, y, verbose=verbose)
    # print("\nTuned Summary (Grid):")
    # reg_tuned_grid.summary()

    # --- Evaluation and Prediction (Using the tuned random model) ---
    print("\n--- Predictions and Evaluations using Tuned (Random) Model ---")
    reg = reg_tuned_random  # Use the tuned model for subsequent steps

    # Print the summary of all models
    # reg.summary() # Already printed above

    # Predict using all models or a specific model
    predictions = reg.predict(X[:3])
    # Note: Model name might change slightly if tuning is complex, but base name should work
    try:
        specific_pred = reg.predict(X[:3], model="RandomForestClassifier")
        print(f"\nRandom Forest Classifier Predictions (Tuned): {specific_pred}")
    except ValueError as e:
        print(f"\nCould not get specific prediction for RandomForestClassifier: {e}")

    print("\nAll Predictions (Tuned):")
    for model, pred in predictions.items():
        print(f"\t{model}: {pred}")

    # Evaluate all or a specific model
    results = reg.evaluate(y)
    try:
        results_specific = reg.evaluate(y, model="RandomForestClassifier")
        print(f"\nRandom Forest Classifier Results (Tuned): {results_specific}")
    except ValueError as e:
        print(f"\nCould not get specific evaluation for RandomForestClassifier: {e}")

    print("\nAll Evaluation Results (Tuned):")
    for model, result in results.items():
        print(f"\t{model}: {result}")


if __name__ == "__main__":
    run_example(verbose=True)  # Set verbose=True to see tuning progress
