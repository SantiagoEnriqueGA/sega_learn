from .optimizers import AdamOptimizer, SGDOptimizer, AdadeltaOptimizer
from .schedulers import lr_scheduler_exp, lr_scheduler_plateau, lr_scheduler_step
from .loss import CrossEntropyLoss, BCEWithLogitsLoss
from .layers import Layer
from .activations import Activation
from .neuralNetwork import NeuralNetwork

__all__ = [
    'AdamOptimizer',
    'SGDOptimizer',
    'AdadeltaOptimizer',
    'lr_scheduler_exp',
    'lr_scheduler_plateau',
    'lr_scheduler_step',
    'CrossEntropyLoss',
    'BCEWithLogitsLoss',
    'NeuralNetwork',
    'Layer',
    'Activation',
]

try:
    from .numba_utils import *
    from .optimizers_jit import JITAdamOptimizer, JITSGDOptimizer, JITAdadeltaOptimizer
    from .layers_jit import JITLayer

    __all__.extend([
        'JITAdamOptimizer',
        'JITSGDOptimizer',
        'JITAdadeltaOptimizer',
        'JITLayer',
    ])
except:
    pass

