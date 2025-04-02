import csv
import os
import sys
import time

import numpy as np

# Set seed for reproducibility
np.random.seed(2)

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sega_learn.neural_networks import *


def time_function(func, num_repeats, *args, **kwargs):
    times = []
    for _ in range(num_repeats):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)
    avg_time = np.mean(times)
    stddev_time = np.std(times)
    return avg_time, stddev_time, result


def time_nn_base(num_repeats=5, layer_sizes_multiplier=5, dataset_size=100_000):
    NUM_REPEATS = num_repeats

    # Neural Network Timing
    # ---------------------------------------------------------------------------------------------
    LAYER_SIZES_MULTIPLIER = layer_sizes_multiplier
    DATASET_SIZE = dataset_size

    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 50, 10]]
    nn = BaseBackendNeuralNetwork(
        layers=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=["relu", "relu", "softmax"],
    )

    X = np.random.randn(DATASET_SIZE, layer_sizes[0])
    y = np.random.randint(0, layer_sizes[-1], size=(DATASET_SIZE,))

    print(
        f"\nTiming results for NeuralNetwork methods (averaged over {NUM_REPEATS} runs):"
    )
    print(
        f"Performance for Layer sizes: {layer_sizes}, Dataset size: {DATASET_SIZE:,} samples"
    )
    print("-" * 100)
    print(f"{'Function':<19} : {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 70)

    nn_times = {}

    # Time forward pass
    forward_avg, forward_stddev, _ = time_function(nn.forward, NUM_REPEATS, X)
    nn_times["forward"] = (forward_avg, forward_stddev)
    print(f"{'forward':<19} : {forward_avg:.6f} seconds ± {forward_stddev:.6f} seconds")

    # Time apply_dropout
    apply_dropout_avg, apply_dropout_stddev, _ = time_function(
        nn.apply_dropout, NUM_REPEATS, X
    )
    nn_times["apply_dropout"] = (apply_dropout_avg, apply_dropout_stddev)
    print(
        f"{'apply_dropout':<19} : {apply_dropout_avg:.6f} seconds ± {apply_dropout_stddev:.6f} seconds"
    )

    # Time backward pass
    nn.forward(X)  # Ensure forward pass is done before backward pass
    backward_avg, backward_stddev, _ = time_function(nn.backward, NUM_REPEATS, y)
    nn_times["backward"] = (backward_avg, backward_stddev)
    print(
        f"{'backward':<19} : {backward_avg:.6f} seconds ± {backward_stddev:.6f} seconds"
    )

    # Time calculate_loss
    calculate_loss_avg, calculate_loss_stddev, _ = time_function(
        nn.calculate_loss, NUM_REPEATS, X, y
    )
    nn_times["calculate_loss"] = (calculate_loss_avg, calculate_loss_stddev)
    print(
        f"{'calculate_loss':<19} : {calculate_loss_avg:.6f} seconds ± {calculate_loss_stddev:.6f} seconds"
    )

    # Time evaluate
    evaluate_avg, evaluate_stddev, _ = time_function(nn.evaluate, NUM_REPEATS, X, y)
    nn_times["evaluate"] = (evaluate_avg, evaluate_stddev)
    print(
        f"{'evaluate':<19} : {evaluate_avg:.6f} seconds ± {evaluate_stddev:.6f} seconds"
    )

    # Time train (single epoch)
    # Use a smaller dataset for training to avoid long training times DATASET_SIZE/10
    X_small = np.random.randn(DATASET_SIZE // 10, layer_sizes[0])
    y_small = np.random.randint(0, layer_sizes[-1], size=(DATASET_SIZE // 10,))

    optimizer = AdamOptimizer(learning_rate=0.01)
    sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=100)
    scheduler = lr_scheduler_plateau(sub_scheduler, patience=100, threshold=0.001)
    train_avg, train_stddev, _ = time_function(
        nn.train,
        NUM_REPEATS,
        X_small,
        y_small,
        epochs=1,
        batch_size=32,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        p=False,
        use_tqdm=False,
    )
    nn_times["train"] = (train_avg, train_stddev)
    print(
        f"{'train (X,y Reduced)':<19} : {train_avg:.6f} seconds ± {train_stddev:.6f} seconds"
    )

    return nn_times


def time_nn_optimizer(num_repeats=5, layer_sizes_multiplier=25):
    # Optimizer Timing
    # ---------------------------------------------------------------------------------------------
    NUM_REPEATS = num_repeats
    LAYER_SIZES_MULTIPLIER = layer_sizes_multiplier

    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 75, 50, 25]]
    nn = BaseBackendNeuralNetwork(
        layers=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=["relu", "relu", "softmax"],
    )

    optimizers = {
        "Adam": AdamOptimizer(
            learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, reg_lambda=0.01
        ),
        "SGD": SGDOptimizer(learning_rate=0.001, momentum=0.9, reg_lambda=0.01),
        "Adadelta": AdadeltaOptimizer(
            learning_rate=0.001, rho=0.95, epsilon=1e-8, reg_lambda=0.01
        ),
    }

    optimizer_times = {}
    for name, optimizer in optimizers.items():
        # Time initialize
        init_avg, init_stddev, _ = time_function(
            optimizer.initialize, NUM_REPEATS, nn.layers
        )
        optimizer_times[f"{name}_initialize"] = (init_avg, init_stddev)

        # Time update
        # Simulate gradients for weights and biases
        dW = [np.random.randn(*layer.weights.shape) for layer in nn.layers]
        db = [np.random.randn(*layer.biases.shape) for layer in nn.layers]
        # Time the update method for each layer
        update_avg, update_stddev, _ = time_function(
            lambda: [
                optimizer.update(layer, dW[i], db[i], i)
                for i, layer in enumerate(nn.layers)
            ],
            NUM_REPEATS,
        )
        optimizer_times[f"{name}_update"] = (update_avg, update_stddev)

    # Print the optimizer timing results
    print(f"\nTiming results for Optimizer methods (averaged over {NUM_REPEATS} runs):")
    print(f"Performance for Layer sizes: {layer_sizes}")
    print("-" * 100)
    print(f"{'Optimizer Function':<25}: {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 70)

    # Print initialize timings
    for func_name, (avg_time, stddev_time) in optimizer_times.items():
        if "initialize" in func_name:
            print(
                f"{func_name:<25}: {avg_time:.6f} seconds ± {stddev_time:.6f} seconds"
            )
    print("-" * 70)
    # Print update timings
    for func_name, (avg_time, stddev_time) in optimizer_times.items():
        if "update" in func_name:
            print(
                f"{func_name:<25}: {avg_time:.6f} seconds ± {stddev_time:.6f} seconds"
            )

    return optimizer_times


def time_nn_loss(num_repeats=5, layer_sizes_multiplier=10, dataset_size=1_000_000):
    # Loss Function Timing
    # ---------------------------------------------------------------------------------------------
    NUM_REPEATS = num_repeats
    LAYER_SIZES_MULTIPLIER = layer_sizes_multiplier
    DATASET_SIZE = dataset_size
    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 50, 10]]

    loss_functions = {
        "CrossEntropyLoss": CrossEntropyLoss(),
        "BCEWithLogitsLoss": BCEWithLogitsLoss(),
    }

    loss_times = {}
    for name, loss_fn in loss_functions.items():
        if name == "CrossEntropyLoss":
            logits = np.random.randn(DATASET_SIZE, layer_sizes[-1])
            targets = np.eye(layer_sizes[-1])[
                np.random.choice(layer_sizes[-1], DATASET_SIZE)
            ]
        else:  # BCEWithLogitsLoss
            logits = np.random.randn(DATASET_SIZE)
            targets = np.random.randint(0, 2, size=(DATASET_SIZE,))

        loss_avg, loss_stddev, _ = time_function(loss_fn, NUM_REPEATS, logits, targets)
        loss_times[name] = (loss_avg, loss_stddev)

    # Print the loss function timing results
    print(f"\nTiming results for Loss functions (averaged over {NUM_REPEATS} runs):")
    print(
        f"Performance for Layer sizes: {layer_sizes}, Dataset size: {DATASET_SIZE:,} samples"
    )
    print("-" * 100)
    print(f"{'Loss Function':<25}: {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 75)
    for func_name, (avg_time, stddev_time) in loss_times.items():
        print(f"{func_name:<25}: {avg_time:.6f} seconds ± {stddev_time:.6f} seconds")

    return loss_times


def combine_timing_results(nn_times, optimizer_times, loss_times):
    # Combine all timing results into a single dictionary and save to a csv file
    # ---------------------------------------------------------------------------------------------
    def save_timing_results(filename, *timing_dicts):
        """Combines multiple timing dictionaries and saves them to a CSV file."""
        combined_results = []

        for timing_dict in timing_dicts:
            for func_name, (avg_time, stddev_time) in timing_dict.items():
                combined_results.append([func_name, avg_time, stddev_time])

        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Function", "Average Time (s)", "Std Dev (s)"])
            writer.writerows(combined_results)

        print(f"\nTiming results saved to {filename}")

    # Combine all results and save to CSV
    output_filename = "tests_performance/nn/nn_timing_results.csv"
    save_timing_results(output_filename, nn_times, optimizer_times, loss_times)


def time_nn_epoch(
    num_repeats=5,
    layer_sizes_multiplier=1,
    dataset_sizes=[1_000, 5_000, 10_000, 50_000, 100_000],
    save_csv=True,
):
    # Time train (single epoch) for increasing dataset size
    # ---------------------------------------------------------------------------------------------
    NUM_REPEATS = num_repeats
    DATASET_SIZES = dataset_sizes
    LAYER_SIZES_MULTIPLIER = layer_sizes_multiplier

    times = []
    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 50, 10]]
    nn = BaseBackendNeuralNetwork(
        layers=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=["relu", "relu", "softmax"],
    )

    print(
        f"\nTiming results for NeuralNetwork train method (averaged over {NUM_REPEATS} runs):"
    )
    print(f"Performance for Layer sizes: {layer_sizes}")
    print("-" * 100)
    print(f"{'Dataset Size':<19} : {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 70)
    for dataset_size in DATASET_SIZES:
        X_small = np.random.randn(dataset_size, layer_sizes[0])
        y_small = np.random.randint(0, layer_sizes[-1], size=(dataset_size,))

        optimizer = AdamOptimizer(learning_rate=0.01)
        sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)
        scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)
        train_avg, train_stddev, _ = time_function(
            nn.train,
            NUM_REPEATS,
            X_small,
            y_small,
            epochs=1,
            batch_size=256,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            p=False,
            use_tqdm=False,
        )
        times.append((dataset_size, train_avg, train_stddev))

        print(
            f"{dataset_size:<20,}: {train_avg:.6f} seconds ± {train_stddev:.6f} seconds"
        )

    if save_csv:
        # Save timing results to CSV
        with open(
            "tests_performance/nn/nn_timing_results_train.csv", mode="w", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["Dataset Size", "Average Time (s)", "Std Dev (s)"])
            writer.writerows(times)
        print(
            "\nTiming results saved to tests_performance/nn/nn_timing_results_train.csv"
        )
    return times


if __name__ == "__main__":
    # combine_timing_results(time_nn_base(), time_nn_optimizer(), time_nn_loss())
    time_nn_base(num_repeats=5)
    time_nn_optimizer(num_repeats=5)
    time_nn_loss(num_repeats=5)
    time_nn_epoch(num_repeats=5, save_csv=False)
