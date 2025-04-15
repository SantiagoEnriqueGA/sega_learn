from .auto import (
    AutoClassifier,
    AutoRegressor,
)
from .clustering import (
    DBSCAN,
    KMeans,
)
from .linear_models import (
    RANSAC,
    Bayesian,
    Lasso,
    LinearDiscriminantAnalysis,
    LogisticRegression,
    OrdinaryLeastSquares,
    PassiveAggressiveRegressor,
    Perceptron,
    QuadraticDiscriminantAnalysis,
    Ridge,
    make_sample_data,
)
from .nearest_neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
)
from .neural_networks import (
    Activation,
    AdadeltaOptimizer,
    AdamOptimizer,
    BaseBackendNeuralNetwork,
    BCEWithLogitsLoss,
    ConvLayer,
    CrossEntropyLoss,
    DenseLayer,
    FlattenLayer,
    HuberLoss,
    MeanAbsoluteErrorLoss,
    MeanSquaredErrorLoss,
    NeuralNetworkBase,
    RNNLayer,
    SGDOptimizer,
    lr_scheduler_exp,
    lr_scheduler_plateau,
    lr_scheduler_step,
)
from .svm import (
    BaseSVM,
    GeneralizedSVC,
    GeneralizedSVR,
    LinearSVC,
    LinearSVR,
    OneClassSVM,
)
from .trees import (
    ClassifierTree,
    ClassifierTreeUtility,
    GradientBoostedClassifier,
    GradientBoostedRegressor,
    IsolationForest,
    IsolationTree,
    IsolationUtils,
    RandomForestClassifier,
    RandomForestRegressor,
    RegressorTree,
    RegressorTreeUtility,
)
from .utils import (
    PCA,
    SMOTE,
    SVD,
    Augmenter,
    DataPrep,
    GridSearchCV,
    Metrics,
    ModelSelectionUtility,
    PolynomialTransform,
    RandomOverSampler,
    RandomSearchCV,
    RandomUnderSampler,
    Scaler,
    VotingRegressor,
    make_blobs,
    make_classification,
    make_regression,
    normalize,
    one_hot_encode,
    train_test_split,
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
    "LogisticRegression",
    "Perceptron",
    "make_data"
    # Trees
    "ClassifierTreeUtility",
    "ClassifierTree",
    "RegressorTreeUtility",
    "RegressorTree",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostedClassifier",
    "GradientBoostedRegressor",
    "IsolationTree",
    "IsolationForest",
    "IsolationUtils",
    # Neural Networks
    "AdamOptimizer",
    "SGDOptimizer",
    "AdadeltaOptimizer",
    "lr_scheduler_exp",
    "lr_scheduler_plateau",
    "lr_scheduler_step",
    "CrossEntropyLoss",
    "BCEWithLogitsLoss",
    "MeanSquaredErrorLoss",
    "MeanAbsoluteErrorLoss",
    "HuberLoss",
    "NeuralNetwork",
    "DenseLayer",
    "FlattenLayer",
    "ConvLayer",
    "RNNLayer",
    "Activation",
    "NeuralNetworkBase",
    "BaseBackendNeuralNetwork",
    # Nearest Neighbors
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    # SVM
    "BaseSVM",
    "LinearSVC",
    "LinearSVR",
    "OneClassSVM",
    "GeneralizedSVR",
    "GeneralizedSVC",
    # Auto
    "AutoRegressor",
    "AutoClassifier",
]

try:
    from .neural_networks.layers_jit import (
        JITConvLayer,
        JITDenseLayer,
        JITFlattenLayer,
        JITRNNLayer,
    )
    from .neural_networks.loss_jit import (
        JITBCEWithLogitsLoss,
        JITCrossEntropyLoss,
        JITHuberLoss,
        JITMeanAbsoluteErrorLoss,
        JITMeanSquaredErrorLoss,
    )
    from .neural_networks.neuralNetworkNumbaBackend import NumbaBackendNeuralNetwork
    from .neural_networks.numba_utils import *
    from .neural_networks.optimizers_jit import (
        JITAdadeltaOptimizer,
        JITAdamOptimizer,
        JITSGDOptimizer,
    )

    __all__.extend(
        [
            "JITAdamOptimizer",
            "JITSGDOptimizer",
            "JITAdadeltaOptimizer",
            "JITBCEWithLogitsLoss",
            "JITCrossEntropyLoss",
            "JITMeanSquaredErrorLoss",
            "JITMeanAbsoluteErrorLoss",
            "JITHuberLoss",
            "JITDenseLayer",
            "JITFlattenLayer",
            "JITConvLayer",
            "JITRNNLayer",
            "NumbaBackendNeuralNetwork",
        ]
    )
except ImportError:
    pass
