import numpy as np
from scipy import linalg

def make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1, 
                   bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, 
                   shuffle=True, coef=False, random_state=None):
    """
    Generate a random regression problem.
    
    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=100
        The number of features.
    n_informative : int, default=10
        The number of informative features, i.e., the number of features used
        to build the linear model used to generate the output.
    n_targets : int, default=1
        The number of regression targets, i.e., the dimension of the y output.
    bias : float, default=0.0
        The bias term in the underlying linear model.
    effective_rank : int or None, default=None
        If not None, the approximate dimension of the data matrix.
    tail_strength : float, default=0.5
        The relative importance of the fat noisy tail of the singular values
        profile if `effective_rank` is not None.
    noise : float, default=0.0
        The standard deviation o    f the gaussian noise applied to the output.
    shuffle : bool, default=True
        Whether to shuffle the samples and the features.
    coef : bool, default=False
        If True, the coefficients of the underlying linear model are returned.
    random_state : int or None, default=None
        Random seed.
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        The output values.
    coef : ndarray of shape (n_features,) or (n_features, n_targets)
        The coefficient of the underlying linear model. Only returned if
        coef=True.
    """
    
    # Set random state
    rng = np.random.RandomState(random_state)
    
    # Generate random design matrix X
    if effective_rank is None:
        # Generate data of full rank
        X = rng.normal(size=(n_samples, n_features))
    else:
        # Generate data of approximate rank `effective_rank`
        if effective_rank > n_features:
            raise ValueError("effective_rank must be less than n_features")
        
        # Create covariance matrix with singular values decreasing exponentially
        singular_values = np.zeros(n_features)
        singular_values[:effective_rank] = np.exp(-np.arange(effective_rank) / effective_rank * tail_strength)
        singular_values[effective_rank:] = singular_values[effective_rank - 1] / 100
        
        # Create random covariance matrix
        U, _, _ = linalg.svd(rng.normal(size=(n_features, n_features)), full_matrices=False)
        X = rng.normal(size=(n_samples, 1)) @ np.ones((1, n_features))
        X = np.dot(X, np.dot(U, np.diag(np.sqrt(singular_values))))
    
    # Generate true coefficients
    if n_informative > n_features:
        n_informative = n_features
    
    ground_truth = np.zeros((n_features, n_targets))
    ground_truth[:n_informative, :] = rng.normal(size=(n_informative, n_targets))
    
    # Build output
    y = np.dot(X, ground_truth) + bias
    
    # Add noise
    if noise > 0.0:
        y += rng.normal(scale=noise, size=y.shape)
    
    # Shuffle
    if shuffle:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    if n_targets == 1:
        y = y.ravel()
    
    if coef:
        return X, y, ground_truth
    else:
        return X, y

def make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2,
                       n_repeated=0, n_classes=2, n_clusters_per_class=2,
                       weights=None, flip_y=0.01, class_sep=1.0, hypercube=True,
                       shift=0.0, scale=1.0, shuffle=True, random_state=None):
    """
    Generate a random n-class classification problem.
    
    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.
    n_features : int, default=20
        The total number of features.
    n_informative : int, default=2
        The number of informative features.
    n_redundant : int, default=2
        The number of redundant features.
    n_repeated : int, default=0
        The number of duplicated features.
    n_classes : int, default=2
        The number of classes (or labels) of the classification problem.
    n_clusters_per_class : int, default=2
        The number of clusters per class.
    weights : array-like of shape (n_classes,) or None, default=None
        The proportions of samples assigned to each class.
    flip_y : float, default=0.01
        The fraction of samples whose class is randomly exchanged.
    class_sep : float, default=1.0
        The factor multiplying the hypercube size.
    hypercube : bool, default=True
        If True, the clusters are put on the vertices of a hypercube.
    shift : float, default=0.0
        Shift features by the specified value.
    scale : float, default=1.0
        Multiply features by the specified value.
    shuffle : bool, default=True
        Shuffle the samples and the features.
    random_state : int or None, default=None
        Random seed.
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.
    y : ndarray of shape (n_samples,)
        The integer labels for class membership of each sample.
    """
    
    # Set random state
    rng = np.random.RandomState(random_state)
    
    # Calculate actual number of features
    n_useless = n_features - n_informative - n_redundant - n_repeated
    if n_useless < 0:
        raise ValueError("n_features must be greater or equal to n_informative + n_redundant + n_repeated")
    
    # Normalize weights
    if weights is not None:
        weights = np.asarray(weights)
        if weights.shape[0] != n_classes:
            raise ValueError("weights must be of length n_classes")
        weights_sum = np.sum(weights)
        if weights_sum <= 0:
            raise ValueError("weights must sum to a positive value")
        weights = weights / weights_sum
    else:
        weights = np.ones(n_classes) / n_classes
    
    # Calculate samples per class
    n_samples_per_class = np.random.multinomial(n_samples, weights)
    
    # Initialize data matrices
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    # --- New: Generate a class center for each class ---
    class_centers = rng.randn(n_classes, n_informative)
    if hypercube:
        class_centers = np.sign(class_centers)
    class_centers *= class_sep

    # Define a scale for cluster offsets (small relative to class_sep)
    offset_scale = 0.1 * class_sep

    start = 0
    for i in range(n_classes):
        n_samples_i = n_samples_per_class[i]
        if n_samples_i <= 0:
            continue
        end = start + n_samples_i
        y[start:end] = i

        # For each class, create cluster centers as offsets from the class center
        centroids = class_centers[i] + rng.randn(n_clusters_per_class, n_informative) * offset_scale

        # Get samples for each cluster
        n_samples_per_cluster = np.random.multinomial(n_samples_i, [1 / n_clusters_per_class] * n_clusters_per_class)
        
        cluster_start = 0
        for j, n_cluster_samples in enumerate(n_samples_per_cluster):
            if n_cluster_samples <= 0:
                continue
            cluster_end = cluster_start + n_cluster_samples
            X_cluster = rng.randn(n_cluster_samples, n_informative)
            X_cluster += centroids[j]
            X[start + cluster_start:start + cluster_end, :n_informative] = X_cluster
            cluster_start = cluster_end
        
        start = end
    
    # Add redundant features (linear combinations of informative ones)
    if n_redundant > 0:
        B = rng.randn(n_informative, n_redundant)
        X[:, n_informative:n_informative + n_redundant] = np.dot(X[:, :n_informative], B)
    
    # Add repeated features (copies of some of the first n_informative+n_redundant columns)
    if n_repeated > 0:
        indices = rng.choice(n_informative + n_redundant, n_repeated, replace=True)
        X[:, n_informative + n_redundant:n_informative + n_redundant + n_repeated] = X[:, indices]
    
    # Add useless features (noise)
    if n_useless > 0:
        X[:, -n_useless:] = rng.randn(n_samples, n_useless)
    
    # Apply shift and scale
    X = X * scale + shift
    
    # Flip labels (ensure changed labels)
    if flip_y > 0.0:
        flip_mask = rng.rand(n_samples) < flip_y
        if n_classes == 2:
            y[flip_mask] = 1 - y[flip_mask]
        else:
            random_labels = rng.randint(0, n_classes, size=np.sum(flip_mask))
            same = random_labels == y[flip_mask]
            random_labels[same] = (y[flip_mask][same] + 1) % n_classes
            y[flip_mask] = random_labels

    # Shuffle samples and features
    if shuffle:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]
        
        feature_indices = np.arange(n_features)
        rng.shuffle(feature_indices)
        X = X[:, feature_indices]
    
    return X, y

def make_blobs(n_samples=100, n_features=2, centers=None, cluster_std=1.0,
              center_box=(-10.0, 10.0), shuffle=True, random_state=None):
    """
    Generate isotropic Gaussian blobs for clustering.
    
    Parameters
    ----------
    n_samples : int or array-like, default=100
        If int, it is the total number of samples.
        If array-like, it contains the number of samples per cluster.
    n_features : int, default=2
        The number of features.
    centers : int or array-like of shape (n_centers, n_features), default=None
        The number of centers to generate, or the fixed center locations.
        If n_samples is an int and centers is None, 3 centers are generated.
        If n_samples is array-like, centers must be
        either None or an array of length equal to the length of n_samples.
    cluster_std : float or array-like of shape (n_centers,), default=1.0
        The standard deviation of the clusters.
    center_box : tuple of float (min, max), default=(-10.0, 10.0)
        The bounding box for each cluster center when centers are
        generated at random.
    shuffle : bool, default=True
        Shuffle the samples.
    random_state : int or None, default=None
        Random seed.
        
    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.
    y : ndarray of shape (n_samples,)
        The integer labels for cluster membership of each sample.
    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster. Only returned if return_centers is True.
    """
    
    # Set random state
    rng = np.random.RandomState(random_state)
    
    # Handle n_samples
    if isinstance(n_samples, (list, tuple)):
        n_samples_per_center = n_samples
        n_centers = len(n_samples_per_center)
        n_samples = sum(n_samples_per_center)
        if centers is None:
            centers = rng.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        else:
            centers = np.asarray(centers)
            if centers.shape[0] != n_centers:
                raise ValueError("centers must have shape (n_centers, n_features)")
    else:
        if centers is None:
            n_centers = 3
            centers = rng.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        elif isinstance(centers, int):
            n_centers = centers
            centers = rng.uniform(center_box[0], center_box[1], size=(n_centers, n_features))
        else:
            centers = np.asarray(centers)
            n_centers = centers.shape[0]
        n_samples_per_center = [n_samples // n_centers] * n_centers
        remainder = n_samples % n_centers
        for i in range(remainder):
            n_samples_per_center[i] += 1
    
    # Handle cluster_std
    if np.isscalar(cluster_std):
        cluster_std = np.ones(n_centers) * cluster_std
    else:
        cluster_std = np.asarray(cluster_std)
    
    # Get n_features from centers
    n_features = centers.shape[1]
    
    # Initialize data matrices
    X = np.zeros((n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    
    # Build data
    start = 0
    for i, (n, std) in enumerate(zip(n_samples_per_center, cluster_std)):
        end = start + n
        X[start:end] = centers[i] + rng.normal(scale=std, size=(n, n_features))
        y[start:end] = i
        start = end
    
    # Shuffle
    if shuffle:
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        X = X[indices]
        y = y[indices]
    
    return X, y, centers