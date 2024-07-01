import numpy as np

class KMeans:
    """
    K-Means clustering algorithm implementation.

    Parameters
    ----------
    k : int
        Number of clusters.
    init : {'random', 'k-means++'}, default='random'
        Method for initialization:
        - 'random': Randomly selects k data points as initial centroids.
        - 'k-means++': Selects centroids using the k-means++ initialization algorithm.
    max_iters : int, default=100
        Maximum number of iterations.
    tol : float, default=1e-4
        Tolerance to declare convergence.

    Attributes
    ----------
    centroids : array, shape (k, n_features)
        Coordinates of cluster centroids.
    labels : array, shape (n_samples,)
        Labels of each point (cluster index) after fitting.
    inertia_ : float
        Sum of squared distances of samples to their closest centroid.

    Methods
    -------
    fit(X)
        Compute k-means clustering on input data X.
    predict(X)
        Predict the closest cluster each sample in X belongs to.
    """

    def __init__(self, k, init='random', max_iters=100, tol=1e-4):
        self.k = k
        self.init = init
        self.max_iters = max_iters
        self.tol = tol

    def fit(self, X):
        """
        Compute k-means clustering.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples, n_features = X.shape

        if self.init == 'random':
            # Randomly initialize centroids by selecting k random samples from the data
            centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        elif self.init == 'k-means++':
            # Initialize centroids using the k-means++ algorithm
            centroids = np.empty((self.k, n_features))
            centroids[0] = X[np.random.choice(n_samples)]
            for i in range(1, self.k):
                distances = np.min(np.linalg.norm(X[:, np.newaxis] - centroids[:i], axis=2), axis=1)
                probs = distances / np.sum(distances)
                centroids[i] = X[np.random.choice(n_samples, p=probs)]
        else:
            raise ValueError("init parameter must be 'random' or 'k-means++'")

        for iteration in range(self.max_iters):
            # Compute distances from each point to each centroid
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

            # Assign each sample to the nearest centroid
            labels = np.argmin(distances, axis=1)

            # Compute new centroids
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.k)])

            # Check for convergence
            if np.linalg.norm(new_centroids - centroids) < self.tol:
                break

            centroids = new_centroids

        self.centroids = centroids
        self.labels = labels
        self.inertia_ = np.sum(np.min(np.linalg.norm(X[:, np.newaxis] - centroids, axis=2), axis=1))

        return self

    def predict(self, X):
        """
        Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)
