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
    train_test_split,
    one_hot_encode,
    normalize,
    Scaler,
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
    GradientBoostedRegressor,
    IsolationTree,
    IsolationForest,    
    IsolationUtils,
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
    
from .svm import (
    BaseSVM,
    LinearSVC,
    LinearSVR,
    OneClassSVM,
    GeneralizedSVR,
    GeneralizedSVC,
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
    "one_hot_encode",
    "normalize",
    "Scaler",
    
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
    "IsolationTree",
    "IsolationForest",
    "IsolationUtils",
    
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
    
    # SVM
    'BaseSVM', 
    'LinearSVC', 
    'LinearSVR', 
    'OneClassSVM',
    'GeneralizedSVR',
    'GeneralizedSVC',
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

