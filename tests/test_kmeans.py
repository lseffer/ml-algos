from kmeans import KMeans
import numpy as np

def test_kmeans():
    X = np.random.normal(size=(50, 2))
    km = KMeans(nr_clusters=2)
    km.fit(X)
    assert km.centroids.shape[0] == 2
    distances = []
    for centroid in km.centroids:
        distances.append(km.euclidean_distance_2d(centroid, X[-1:]))
    assert km.predict(X[-1:])[0] == np.argmin(distances)
