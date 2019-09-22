import numpy as np

class KMeans():

    def __init__(self, nr_clusters=3, iterations=10000):
        self.nr_clusters = nr_clusters
        self.iterations = iterations

    def euclidean_distance_2d(self, centroid: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.sqrt(((X - centroid) ** 2).sum(axis=1))

    def closest_centroid(self, X: np.ndarray) -> np.ndarray:
        tmp_arr = np.zeros(shape=(X.shape[0], self.nr_clusters))
        for idx, centroid in enumerate(self.centroids):
            distance = self.euclidean_distance_2d(centroid, X)
            tmp_arr[:, idx] = distance
        return np.argmin(tmp_arr, axis=1)

    def update_centroids(self, X: np.ndarray, closest_centroids: np.ndarray) -> np.ndarray:
        for idx, centroid in enumerate(self.centroids):
            self.centroids[idx] = np.average(X[closest_centroids == idx, :], axis=0)

    def fit(self, X: np.ndarray) -> None:
        self.centroids = X[np.random.randint(low=0, high=X.shape[0], size=(self.nr_clusters, )), :]
        for iteration in range(self.iterations):
            closest_centroids = self.closest_centroid(X)
            self.update_centroids(X, closest_centroids)

    def predict(self, X):
        return self.closest_centroid(X)
