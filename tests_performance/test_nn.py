import os
import sys
import time
import numpy as np

from utils import suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

def main():
    NUM_REPEATS = 5
    
    # Neural Network Timing
    # ---------------------------------------------------------------------------------------------
    LAYER_SIZES_MULTIPLIER = 5
    DATASET_SIZE = 100_000
    
    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 50, 10]]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=['relu', 'relu', 'softmax'],
    )
    
    X = np.random.randn(DATASET_SIZE, layer_sizes[0])
    y = np.random.randint(0, layer_sizes[-1], size=(DATASET_SIZE,))

    nn_times = {}
    
    # Time forward pass
    forward_avg, forward_stddev, _ = time_function(nn.forward, NUM_REPEATS, X)
    nn_times['forward'] = (forward_avg, forward_stddev)

    # Time apply_dropout
    apply_dropout_avg, apply_dropout_stddev, _ = time_function(nn.apply_dropout, NUM_REPEATS, X)
    nn_times['apply_dropout'] = (apply_dropout_avg, apply_dropout_stddev)

    # Time backward pass
    nn.forward(X)  # Ensure forward pass is done before backward pass
    backward_avg, backward_stddev, _ = time_function(nn.backward, NUM_REPEATS, y)
    nn_times['backward'] = (backward_avg, backward_stddev)

    # Time calculate_loss
    calculate_loss_avg, calculate_loss_stddev, _ = time_function(nn.calculate_loss, NUM_REPEATS, X, y)
    nn_times['calculate_loss'] = (calculate_loss_avg, calculate_loss_stddev)

    # Time evaluate
    evaluate_avg, evaluate_stddev, _ = time_function(nn.evaluate, NUM_REPEATS, X, y)
    nn_times['evaluate'] = (evaluate_avg, evaluate_stddev)

    # Time train (single epoch)
    # Use a smaller dataset for training to avoid long training times DATASET_SIZE/10
    X_small = np.random.randn(DATASET_SIZE // 10, layer_sizes[0])
    y_small = np.random.randint(0, layer_sizes[-1], size=(DATASET_SIZE // 10,))
    
    optimizer = AdamOptimizer(learning_rate=0.01)
    sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)  
    scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)  
    train_avg, train_stddev, _ = time_function(nn.train, 
                                               NUM_REPEATS, 
                                               X_small, y_small,
                                               epochs=1, batch_size=32, 
                                               optimizer=optimizer, lr_scheduler=scheduler, 
                                               p=False)
    nn_times['train'] = (train_avg, train_stddev)

    # Print the timing results
    print(f"\nTiming results for NeuralNetwork methods (averaged over {NUM_REPEATS} runs):")
    print(f"Performance for Layer sizes: {layer_sizes}, Dataset size: {DATASET_SIZE:,} samples")
    print("-" * 100)    
    print(f"{'Function':<19} : {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 70)
    for func_name, (avg_time, stddev_time) in nn_times.items():
        if func_name == 'train':
            func_name = 'train (X,y Reduced)'
            
        print(f"{func_name:<20}: {avg_time:.6f} seconds ± {stddev_time:.6f} seconds")

    
    # Optimizer Timing
    # ---------------------------------------------------------------------------------------------
    LAYER_SIZES_MULTIPLIER = 25
    
    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 75, 50, 25]]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=['relu', 'relu', 'softmax'],
    )

    optimizers = {
        'Adam': AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, reg_lambda=0.01),
        'SGD': SGDOptimizer(learning_rate=0.001, momentum=0.9, reg_lambda=0.01),
        'Adadelta': AdadeltaOptimizer(learning_rate=0.001, rho=0.95, epsilon=1e-8, reg_lambda=0.01),
    }

    optimizer_times = {}
    for name, optimizer in optimizers.items():
        # Time initialize
        init_avg, init_stddev, _ = time_function(optimizer.initialize, NUM_REPEATS, nn.layers)
        optimizer_times[f'{name}_initialize'] = (init_avg, init_stddev)

        # Time update
        # Simulate gradients for weights and biases
        dW = [np.random.randn(*layer.weights.shape) for layer in nn.layers]
        db = [np.random.randn(*layer.biases.shape) for layer in nn.layers]
        # Time the update method for each layer
        update_avg, update_stddev, _ = time_function(lambda: [optimizer.update(layer, dW[i], db[i], i) for i, layer in enumerate(nn.layers)], NUM_REPEATS)
        optimizer_times[f'{name}_update'] = (update_avg, update_stddev)

    # Print the optimizer timing results
    print(f"\n\nTiming results for Optimizer methods (averaged over {NUM_REPEATS} runs):")
    print(f"Performance for Layer sizes: {layer_sizes}")
    print("-" * 100)    
    print(f"{'Optimizer Function':<25}: {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 70)
    
    # Print initialize timings
    for func_name, (avg_time, stddev_time) in optimizer_times.items():
        if 'initialize' in func_name:
            print(f"{func_name:<25}: {avg_time:.6f} seconds ± {stddev_time:.6f} seconds")
    print("-" * 70)
    # Print update timings
    for func_name, (avg_time, stddev_time) in optimizer_times.items():
        if 'update' in func_name:
            print(f"{func_name:<25}: {avg_time:.6f} seconds ± {stddev_time:.6f} seconds")
    
    
    # Loss Function Timing
    # ---------------------------------------------------------------------------------------------
    LAYER_SIZES_MULTIPLIER = 10
    DATASET_SIZE = 1_000_000
    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 50, 10]]

    loss_functions = {
        'CrossEntropyLoss': CrossEntropyLoss(),
        'BCEWithLogitsLoss': BCEWithLogitsLoss()
    }

    loss_times = {}
    for name, loss_fn in loss_functions.items():
        if name == 'CrossEntropyLoss':
            logits = np.random.randn(DATASET_SIZE, layer_sizes[-1])
            targets = np.eye(layer_sizes[-1])[np.random.choice(layer_sizes[-1], DATASET_SIZE)]
        else:  # BCEWithLogitsLoss
            logits = np.random.randn(DATASET_SIZE)
            targets = np.random.randint(0, 2, size=(DATASET_SIZE,))

        loss_avg, loss_stddev, _ = time_function(loss_fn, NUM_REPEATS, logits, targets)
        loss_times[name] = (loss_avg, loss_stddev)

    # Print the loss function timing results
    print(f"\n\nTiming results for Loss functions (averaged over {NUM_REPEATS} runs):")
    print(f"Performance for Layer sizes: {layer_sizes}, Dataset size: {DATASET_SIZE:,} samples")
    print("-" * 100)    
    print(f"{'Loss Function':<25}: {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 75)
    for func_name, (avg_time, stddev_time) in loss_times.items():
        print(f"{func_name:<25}: {avg_time:.6f} seconds ± {stddev_time:.6f} seconds")


    # Combine all timing results into a single dictionary and save to a csv file
    # ---------------------------------------------------------------------------------------------
    import csv

    def save_timing_results(filename, *timing_dicts):
        """Combines multiple timing dictionaries and saves them to a CSV file."""
        combined_results = []
        
        for timing_dict in timing_dicts:
            for func_name, (avg_time, stddev_time) in timing_dict.items():
                combined_results.append([func_name, avg_time, stddev_time])
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Function", "Average Time (s)", "Std Dev (s)"])
            writer.writerows(combined_results)

        print(f"\nTiming results saved to {filename}")

    # Combine all results and save to CSV
    output_filename = "tests_performance/nn_timing_results.csv"
    save_timing_results(output_filename, nn_times, optimizer_times, loss_times)


    # Time train (single epoch) for increasing dataset size
    # ---------------------------------------------------------------------------------------------
    times = []
    NUM_REPEATS = 5
    DATASET_SIZES = [1_000, 5_000, 10_000, 50_000, 100_000]
    LAYER_SIZES_MULTIPLIER = 1

    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 50, 10]]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=['relu', 'relu', 'softmax'],
    )

    for dataset_size in DATASET_SIZES:
        print(f"Training on dataset size {dataset_size:,} samples")

        X_small = np.random.randn(dataset_size, layer_sizes[0])
        y_small = np.random.randint(0, layer_sizes[-1], size=(dataset_size,))
        
        optimizer = AdamOptimizer(learning_rate=0.01)
        sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)  
        scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)  
        train_avg, train_stddev, _ = time_function(nn.train, 
                                                   NUM_REPEATS, 
                                                   X_small, y_small,
                                                   epochs=1, batch_size=32, 
                                                   optimizer=optimizer, lr_scheduler=scheduler, 
                                                   p=False)
        times.append((dataset_size, train_avg, train_stddev))
    
    # Save timing results to CSV
    with open("tests_performance/nn_timing_results_train.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset Size", "Average Time (s)", "Std Dev (s)"])
        writer.writerows(times)
    print(f"\nTiming results saved to tests_performance/nn_timing_results_train.csv")


if __name__ == "__main__":
    main()

