# Clustering Module

The clustering module in SEGA_LEARN provides implementations of popular clustering algorithms such as KMeans and DBSCAN. These algorithms are designed to group similar data points together based on their features.

## Algorithms

### KMeans
KMeans is a clustering algorithm that partitions n data points into k clusters in which each point belongs to the cluster with the nearest mean.

#### Algorithm
1. Initialize k centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update the centroids by calculating the mean of all data points assigned to each centroid.
4. Repeat steps 2 and 3 until convergence (i.e., the centroids do not change significantly).

#### Usage
```python
from sega_learn.clustering import KMeans

# Initialize the KMeans object
kmeans = KMeans(X, n_clusters=3, max_iter=300, tol=1e-4)

# Fit the model
kmeans.fit()

# Predict cluster labels for new data
labels = kmeans.predict(new_X)

# Find the optimal number of clusters
ch_optimal_k, db_optimal_k, silhouette_optimal_k = kmeans.find_optimal_clusters(max_k=10)
```

### DBSCAN
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a clustering algorithm that groups together points that are closely packed, marking as outliers points that lie alone in low-density regions.

#### Algorithm
1. For each point in the dataset, find the points within a distance $` \epsilon `$ (eps).
2. If the number of points within $` \epsilon `$ is greater than or equal to min_samples, mark the point as a core point
3. Form clusters by connecting core points and their neighbors.
4. Mark points that are not reachable from any core point as outliers.

#### Usage
```python
from sega_learn.clustering import DBSCAN

# Initialize the DBSCAN object
dbscan = DBSCAN(X, eps=0.5, min_samples=5)

# Fit the model
dbscan.fit()

# Predict cluster labels for new data
labels = dbscan.predict(new_X)

# Find the optimal eps value
optimal_eps = dbscan.auto_eps(min=0.1, max=1.1, precision=0.01)
```

## Distance Formulas
Both KMeans and DBSCAN algorithms use distance metrics to measure the similarity between data points. The following distance formulas are available in the clustering module:

### Euclidean Distance
The Euclidean distance is the straight-line distance between two points in Euclidean space. It is calculated as the square root of the sum of squared differences between corresponding coordinates. It is the most common distance metric used in clustering algorithms.

#### Formula
```math
d(x_i, x_j) = \sqrt{\sum_{m=1}^{M} (x_{im} - x_{jm})^2}
```

### Manhattan Distance
The Manhattan distance is the sum of the absolute differences between corresponding coordinates of two points. It is also known as the taxicab or city block distance. It is less sensitive to outliers than the Euclidean distance. It is generally used in clustering algorithms when the features have different scales.

#### Formula
```math
d(x_i, x_j) = \sum_{m=1}^{M} |x_{im} - x_{jm}|
```

### Cosine Distance
The cosine distance is a measure of similarity between two vectors based on the cosine of the angle between them. It is often used in text mining and information retrieval to compare documents based on their term frequency vectors. It is particularly useful when the magnitude of the vectors is not important.

#### Formula
```math
d(x_i, x_j) = 1 - \frac{x_i \cdot x_j}{||x_i|| \cdot ||x_j||}
```

## Evaluation Metrics

### Elbow Method
The elbow method is used to determine the optimal number of clusters by plotting the distortion for different values of k and selecting the k at the "elbow" point. It is based on the intuition that the distortion decreases as k increases, but the rate of decrease slows down after a certain point. Generally, the optimal number of clusters is chosen at the "elbow" point where the distortion starts to flatten out.

#### Formula
The distortion is calculated as the sum of squared distances between each data point and its assigned centroid:
```math
\text{Distortion} = \sum_{i=1}^{N} \min_{j} d(x_i, \mu_j)^2
```

### Calinski-Harabasz Index
The Calinski-Harabasz Index evaluates the clustering performance by measuring the ratio of the sum of between-cluster dispersion to the sum of within-cluster dispersion. A higher CH Index indicates better clustering performance. It is based on the intuition that clusters should be well-separated and compact.

#### Formula
```math
\text{CH Index} = \frac{\sum_{i=1}^{k} n_i \|m_i - m\|^2}{\sum_{i=1}^{k} \sum_{j=1}^{n_i} \|x_{ij} - m_i\|^2} \times \frac{N - k}{k - 1}
```
where $` k `$ is the number of clusters, $` n_i `$ is the number of data points in cluster $` i `$, $` m_i `$ is the centroid of cluster $` i `$, $` m `$ is the overall centroid of all data points, $` x_{ij} $` is the $` j `$-th data point in cluster $` i `$, $` N `$ is the total number of data points.

### Davies-Bouldin Index
The Davies-Bouldin Index evaluates the clustering performance by measuring the average similarity ratio of each cluster with its most similar cluster. A lower DB Index indicates better clustering performance. It is based on the intuition that clusters should be well-separated and compact. The DB Index is sensitive to the number of clusters and the distance metric used.

#### Formula
```math
\text{DB Index} = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \left( \frac{s_i + s_j}{d(\mu_i, \mu_j)} \right)
```
where $` s_i `$ is the average distance between each point in cluster $` i `$ and the centroid $` \mu_i `$, and $` d(\mu_i, \mu_j) `$ is the distance between centroids $` \mu_i `$ and $` \mu_j `$.

### Silhouette Score
The Silhouette Score evaluates the clustering performance by measuring how similar a data point is to its own cluster compared to other clusters. It ranges from -1 to 1, where a higher score indicates better clustering performance. It is based on the intuition that clusters should be well-separated and compact. The Silhouette Score is sensitive to the number of clusters and the distance metric used.

#### Formula
For each data point $` i `$:
```math
s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}
```
where $` a(i) `$ is the average distance between $` i `$ and all other points in the same cluster, and $` b(i) `$ is the average distance between $` i `$ and all points in the nearest cluster.

## Examples

### KMeans Example
```python
from sega_learn.clustering import KMeans
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)

# Initialize and fit KMeans
kmeans = KMeans(X, n_clusters=3)
kmeans.fit()

# Predict cluster labels
labels = kmeans.predict(X)

# Print cluster labels
print(labels)
```

### DBSCAN Example
```python
from sega_learn.clustering import DBSCAN
import numpy as np

# Generate sample data
X = np.random.rand(100, 2)

# Initialize and fit DBSCAN
dbscan = DBSCAN(X, eps=0.3, min_samples=5)
dbscan.fit()

# Predict cluster labels
labels = dbscan.predict(X)

# Print cluster labels
print(labels)
```
