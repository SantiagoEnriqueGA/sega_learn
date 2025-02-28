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
# from sega_learn.neural_networks_numba_dev import *

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

def init_compile():
    """
    Numba JIT code compilation can be slow, especially for the first call.
    This function runs a dummy epoch to compile the JIT code.
    """
    print("Compiling Numba JIT code...")
    compile_time = time.time()
    
    # Initialize a small neural network for the purpose of JIT compilation
    print(f"   ...Compiling small neural network", end="", flush=True)
    layer_sizes = [1, 1, 1]
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        dropout_rate=0.5,
        reg_lambda=0.01,
        activations=['relu', 'relu', 'softmax'],
    )
    print(f": Time: {time.time() - compile_time:.2f} seconds")
    
    X = np.random.randn(10, layer_sizes[0])
    y = np.random.randint(0, layer_sizes[-1], size=(10,))

    # Simulate gradients for weights and biases
    layer = nn.layers[0]
    dW1 = np.random.randn(*layer.weights.shape)
    db1 = np.random.randn(*layer.biases.shape)
    
    # Logits and targets for loss functions
    logits = np.random.randn(10, layer_sizes[-1])
    targets = np.eye(layer_sizes[-1])[np.random.choice(layer_sizes[-1], 10)]
    
    # Initialize all optimizers and functions
    print(f"   ...Compiling optimizers", end="", flush=True)
    optimizer = AdamOptimizer(learning_rate=0.01)
    optimizer.initialize(nn.layers)
    optimizer.update(layer, dW1, db1, 0)
    optimizer = SGDOptimizer(learning_rate=0.01, momentum=0.9)
    optimizer.initialize(nn.layers)
    optimizer.update(layer, dW1, db1, 0)
    optimizer= AdadeltaOptimizer(learning_rate=0.01, rho=0.95, epsilon=1e-8)
    optimizer.initialize(nn.layers)
    optimizer.update(layer, dW1, db1, 0)
    print(f": Time: {time.time() - compile_time:.2f} seconds")
    
    # Not converted to Numba JIT yet
    # # Initialize all schedulers
    # print(f"Initializing schedulers...", end="", flush=True)
    # scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)
    # scheduler = lr_scheduler_exp(optimizer, lr_decay=0.1, lr_decay_epoch=10)
    # scheduler = lr_scheduler_plateau(scheduler, patience=10, threshold=0.001)
    
    # Initialize the loss functions
    print(f"   ...Compiling loss functions", end="", flush=True)
    loss_fn = CrossEntropyLoss()
    loss_fn.calculate_loss(logits, targets)
    loss_fn = BCEWithLogitsLoss()
    loss_fn.calculate_loss(logits, targets.reshape(-1, 1))
    print(f": Time: {time.time() - compile_time:.2f} seconds")
    
    
    
    # Run a dummy training step to compile the JIT code
    print(f"   ...Compiling training step", end="", flush=True)
    optimizer = AdamOptimizer(learning_rate=0.01)
    scheduler = lr_scheduler_step(optimizer, lr_decay=0.1, lr_decay_epoch=10)
    nn.forward(X, training=True)
    nn.backward(y)
    nn.train(X, y, epochs=1, batch_size=32, optimizer=optimizer, lr_scheduler=scheduler, p=False, use_tqdm=False)
    print(f": Time: {time.time() - compile_time:.2f} seconds")
    
    end_time = time.time()
    print(f"\nNumba JIT code compilation completed in {end_time - compile_time:.2f} seconds.")

def time_nn_epoch(num_repeats=5, layer_sizes_multiplier=1, dataset_size=100_000):
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

    # X = np.random.randn(dataset_size, layer_sizes[0])
    # y = np.random.randint(0, layer_sizes[-1], size=(dataset_size,))

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
    # init_compile()
    # time_nn_epoch(num_repeats=5)

    import cProfile, pstats
    cProfile.run('time_nn_epoch(num_repeats=5)', 'profile.stats')
    p = pstats.Stats('profile.stats')
    p.sort_stats('cumtime').print_stats(25)
