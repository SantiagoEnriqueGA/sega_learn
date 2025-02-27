import os
import sys
import time
import csv
import numpy as np
import cupy as cp

# Set seed for reproducibility
np.random.seed(2)

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from sega_learn.neural_networks_cupy_dev import *

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


def time_nn_epoch_cupy(num_repeats=5, layer_sizes_multiplier=1, dataset_size=100_000, save_csv=True):
    # Time train (single epoch) for increasing dataset size
    # ---------------------------------------------------------------------------------------------
    NUM_REPEATS = num_repeats
    DATASET_SIZE = dataset_size
    LAYER_SIZES_MULTIPLIER = layer_sizes_multiplier

    times = []
    layer_sizes = [size * LAYER_SIZES_MULTIPLIER for size in [100, 50, 10]]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=['relu', 'relu', 'softmax'],
    )

    X = cp.random.randn(dataset_size, layer_sizes[0])
    y = cp.random.randint(0, layer_sizes[-1], size=(dataset_size,))
    
    optimizer = AdamOptimizer(learning_rate=0.01)
    sub_scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)  
    scheduler = lr_scheduler_plateau(sub_scheduler, patience=5, threshold=0.001)  
    train_avg, train_stddev, _ = time_function(nn.train, 
                                                NUM_REPEATS, 
                                                X, y,
                                                epochs=1, batch_size=256, 
                                                optimizer=optimizer, lr_scheduler=scheduler, 
                                                p=False, use_tqdm=False)
    
    print(f"\nTiming results for NeuralNetwork train method (averaged over {NUM_REPEATS} runs):")
    print(f"Performance for Layer sizes: {layer_sizes}")
    print("-" * 100)
    print(f"{'Dataset Size':<19} : {'Average Time (s)':<16} {'±  Std Dev (s)'}")
    print("=" * 70)

    print(f"{dataset_size:<20,}: {train_avg:.6f} seconds ± {train_stddev:.6f} seconds")
    
    return times


if __name__ == "__main__":
    time_nn_epoch_cupy(num_repeats=10, save_csv=False)

    # import cProfile, pstats
    # cProfile.run('time_nn_epoch_cupy()', 'profile.stats')
    # p = pstats.Stats('profile.stats')
    # p.sort_stats('cumtime').print_stats(25)
