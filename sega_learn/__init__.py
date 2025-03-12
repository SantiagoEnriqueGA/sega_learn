from .utils import (
    PolynomialTransform,
    DataPrep,
    VotingRegressor,
    ModelSelectionUtility,
    GridSearchCV,
    RandomSearchCV,
    Metrics,
    PCA,
    SVD,
    RandomOverSampler,
    RandomUnderSampler,
    SMOTE,
    Augmenter,
    make_blobs,
    make_regression,
    make_classification,
)

from .clustering import (
    KMeans,
    DBSCAN,
)

from .linear_models import (
    OrdinaryLeastSquares,
    Ridge,
    Lasso,
    Bayesian,
    RANSAC,
    PassiveAggressiveRegressor,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    make_sample_data
)

from .trees import (
    ClassifierTreeUtility,
    ClassifierTree,
    RegressorTreeUtility,
    RegressorTree,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostedRegressor
)

from .neural_networks import (
    AdamOptimizer,
    SGDOptimizer,
    AdadeltaOptimizer,    
    lr_scheduler_exp,
    lr_scheduler_plateau,
    lr_scheduler_step,
    CrossEntropyLoss,
    BCEWithLogitsLoss,
    NeuralNetworkBase,
    BaseBackendNeuralNetwork,
    DenseLayer,
    FlattenLayer,
    ConvLayer,
    RNNLayer,
    Activation,
)

from .nearest_neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
    
__all__ = [
    # Utils
    "PolynomialTransform",
    "DataPrep",
    "VotingRegressor",
    "ModelSelectionUtility",
    "GridSearchCV",
    "RandomSearchCV",
    "Metrics",
    "PCA",
    "SVD",
    "RandomOverSampler",
    "RandomUnderSampler",
    "SMOTE",
    "Augmenter",
    "make_blobs",
    "make_regression",
    "make_classification",
    "train_test_split",
    
    # Clustering
    "KMeans",
    "DBSCAN",
    
    # Linear Models
    "OrdinaryLeastSquares",
    "Ridge",
    "Lasso",
    "Bayesian",
    "RANSAC",
    "PassiveAggressiveRegressor",
    "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
    "make_data"
    
    # Trees
    "ClassifierTreeUtility",
    "ClassifierTree",
    "RegressorTreeUtility",
    "RegressorTree",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostedRegressor",
    
    # Neural Networks
    'AdamOptimizer',
    'SGDOptimizer',
    'AdadeltaOptimizer',
    'lr_scheduler_exp',
    'lr_scheduler_plateau',
    'lr_scheduler_step',
    'CrossEntropyLoss',
    'BCEWithLogitsLoss',
    'NeuralNetwork',
    'DenseLayer',
    'FlattenLayer',
    'ConvLayer',
    'RNNLayer',
    'Activation',
    'NeuralNetworkBase',
    'BaseBackendNeuralNetwork',
    
    # Nearest Neighbors
    "KNeighborsClassifier",
    "KNeighborsRegressor",
]

try:
    from .neural_networks.numba_utils import *
    from .neural_networks.optimizers_jit import JITAdamOptimizer, JITSGDOptimizer, JITAdadeltaOptimizer
    from .neural_networks.loss_jit import JITBCEWithLogitsLoss, JITCrossEntropyLoss
    from .neural_networks.layers_jit import JITDenseLayer, JITFlattenLayer, JITConvLayer, JITRNNLayer
    from .neural_networks.neuralNetworkNumbaBackend import NumbaBackendNeuralNetwork
    
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

