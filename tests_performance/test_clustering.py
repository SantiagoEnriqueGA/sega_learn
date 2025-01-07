import unittest
import os
import sys
import time

from test_utils import suppress_print

# Change the working directory to the parent directory to allow importing the segadb package.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sega_learn.clustering import *
from test_utils import suppress_print

from sklearn.datasets import make_blobs
from sklearn import cluster

NUM_SAMPLES = 100_000
NUM_K = 5
NUM_RUNS = 5
MAX_ITER = 100
    
class TestClusteringPerformance(unittest.TestCase):
    def setUp(self):
        self.X, self.y = make_blobs(n_samples=NUM_SAMPLES, n_features=5, centers=NUM_K, cluster_std=0.60, random_state=1)

    def test_kmeans(self):
        def sega_kmeans():
            start_time = time.time()
            kmeans = KMeans(self.X, n_clusters=NUM_K, max_iter=MAX_ITER)
            kmeans.fit()
            kmeans.predict(self.X)            
            end_time = time.time()
            return end_time - start_time
        
        def sklearn_kmeans():
            start_time = time.time()
            kmeans = cluster.KMeans(n_clusters=NUM_K, max_iter=MAX_ITER)
            kmeans.fit(self.X)
            kmeans.predict(self.X)
            end_time = time.time()
            return end_time - start_time
        
        sklearn_times = [sklearn_kmeans() for _ in range(NUM_RUNS)]
        sega_times = [sega_kmeans() for _ in range(NUM_RUNS)]
        
        avg_sklearn_time = sum(sklearn_times) / NUM_RUNS
        avg_sega_time = sum(sega_times) / NUM_RUNS
        
        print(f"\nKMeans performance on {NUM_SAMPLES:,} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn KMeans: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega KMeans: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10     
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
            
    def setUp_dbscan(self):
        self.X, self.y = make_blobs(n_samples=NUM_SAMPLES // 10, n_features=5, centers=NUM_K, cluster_std=0.60, random_state=1)
        
    def test_dbscan(self):
        self.setUp_dbscan()
        
        def sega_dbscan():
            start_time = time.time()
            dbscan = DBSCAN(self.X, eps=0.5, min_samples=5)
            dbscan.fit()
            end_time = time.time()
            return end_time - start_time
        
        def sklearn_dbscan():
            start_time = time.time()
            dbscan = cluster.DBSCAN(eps=0.5, min_samples=5)
            dbscan.fit(self.X)
            end_time = time.time()
            return end_time - start_time
        
        sklearn_times = [sklearn_dbscan() for _ in range(NUM_RUNS)]
        sega_times = [sega_dbscan() for _ in range(NUM_RUNS)]
        
        avg_sklearn_time = sum(sklearn_times) / NUM_RUNS
        avg_sega_time = sum(sega_times) / NUM_RUNS
        
        print(f"\nDBSCAN performance on {NUM_SAMPLES//10:,.0f} samples (average over {NUM_RUNS} runs):")
        print(f"\tsklearn DBSCAN: {avg_sklearn_time:7.4f} seconds")
        print(f"\tsega DBSCAN: {avg_sega_time:10.4f} seconds")
        
        if avg_sklearn_time == 0: avg_sklearn_time = 1e-10
        if avg_sega_time == 0: avg_sega_time = 1e-10     
        if avg_sklearn_time == avg_sega_time:
            print(f"\tSpeed difference, is negligible")     
        elif avg_sklearn_time > avg_sega_time:
            print(f"\tSpeed difference, sega is {avg_sklearn_time / avg_sega_time:.2f}x faster")
        elif avg_sega_time > avg_sklearn_time:
            print(f"\tSpeed difference, sklearn is {avg_sega_time / avg_sklearn_time:.2f}x faster")
    
                
if __name__ == '__main__':
    unittest.main()