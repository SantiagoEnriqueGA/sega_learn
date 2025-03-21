from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .layers import DenseLayer, FlattenLayer,ConvLayer, RNNLayer
from .activations import Activation
from .neuralNetworkBase import NeuralNetworkBase
from .neuralNetworkBaseBackend import BaseBackendNeuralNetwork

__all__ = [
    'AdamOptimizer',
    'SGDOptimizer',
    'AdadeltaOptimizer',
    'lr_scheduler_exp',
    'lr_scheduler_plateau',
    'lr_scheduler_step',
    'CrossEntropyLoss',
    'BCEWithLogitsLoss',
    'DenseLayer',
    'FlattenLayer',
    'ConvLayer',
    'RNNLayer',
    'Activation',
    'NeuralNetworkBase',
    'BaseBackendNeuralNetwork',
]

try:
    from .numba_utils import *
    from .optimizers_jit import JITAdamOptimizer, JITSGDOptimizer, JITAdadeltaOptimizer
    from .loss_jit import JITBCEWithLogitsLoss, JITCrossEntropyLoss
    from .layers_jit import JITDenseLayer, JITFlattenLayer, JITConvLayer, JITRNNLayer
    from .neuralNetworkNumbaBackend import NumbaBackendNeuralNetwork

    __all__.extend([
        'JITAdamOptimizer',
        'JITSGDOptimizer',
        'JITAdadeltaOptimizer',
        'JITBCEWithLogitsLoss',
        'JITCrossEntropyLoss',
        'JITDenseLayer',
        'JITFlattenLayer',
        'JITConvLayer',
        'JITRNNLayer',
        'NumbaBackendNeuralNetwork',
    ])
except:
    pass

try:
    from .neuralNetworkCuPyBackend import CuPyBackendNeuralNetwork
    from .layers_cupy import CuPyActivation, CuPyDenseLayer
    from .optimizers_cupy import CuPyAdamOptimizer, CuPySGDOptimizer, CuPyAdadeltaOptimizer
    from .loss_cupy import CuPyBCEWithLogitsLoss, CuPyCrossEntropyLoss

    __all__.extend([
        'CuPyBackendNeuralNetwork',
        'CuPyActivation',
        'CuPyDenseLayer',
        'CuPyAdamOptimizer',
        'CuPySGDOptimizer',
        'CuPyAdadeltaOptimizer',
        'CuPyBCEWithLogitsLoss',        
        'CuPyCrossEntropyLoss', 
        'CuPyDenseLayer',
    ])
except:
    pass