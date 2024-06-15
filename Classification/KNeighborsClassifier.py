import numpy as np

class DistanceMetrics:
    def __init__(self, distance_metrics='euclidean_distances'):
        """
        Initialize DistanceMetrics object.

        Parameters
        ----------
        distance_metrics : str, default='euclidean_distances'
            The name of the distance metric to use.
            Supported metrics: 'euclidean_distances', 'manhattan_distances', 'squared_euclidean_distances', 'cosine_distances'.
        """
        self.distance_metrics = distance_metrics

    def euclidean_distances(self, X, Y):
        """
        Calculate Euclidean distances between rows of X and Y.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples_X, n_features).

        Y : np.ndarray
            Array of shape (n_samples_Y, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples_X,) with Euclidean distances.
        """
        return np.sqrt(np.sum((X - Y) ** 2, axis=1))

    def manhattan_distances(self, X, Y):
        """
        Calculate Manhattan distances between rows of X and Y.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples_X, n_features).

        Y : np.ndarray
            Array of shape (n_samples_Y, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples_X,) with Manhattan distances.
        """
        return np.sum(np.abs(X - Y), axis=1)

    def squared_euclidean_distances(self, X, Y):
        """
        Calculate squared Euclidean distances between rows of X and Y.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples_X, n_features).

        Y : np.ndarray
            Array of shape (n_samples_Y, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples_X,) with squared Euclidean distances.
        """
        return np.sum((X - Y) ** 2, axis=1)

    def cosine_distances(self, X, Y):
        """
        Calculate cosine distances (1 - cosine similarity) between rows of X and Y.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples_X, n_features).

        Y : np.ndarray
            Array of shape (n_samples_Y, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples_X,) with cosine distances.
        """
        cos_sim = np.dot(X, Y) / (np.linalg.norm(X, axis=1) * np.linalg.norm(Y))
        return 1 - cos_sim

    def __call__(self, X, Y):
        """
        Compute distances between X and Y using the specified distance metric.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples_X, n_features).

        Y : np.ndarray
            Array of shape (n_samples_Y, n_features).

        Returns
        -------
        np.ndarray
            Array of shape (n_samples_X,) with distances computed using the specified metric.

        Raises
        ------
        ValueError
            If an invalid distance metric is provided.
        """
        try:
            return getattr(self, self.distance_metrics)(X, Y)
        except AttributeError:
            raise ValueError(f"Invalid distance metric: {self.distance_metrics}")


class KNeighborsClassifier:
    def __init__(self, n_neighbors=5, weights='uniform', distance_metrics='euclidean_distances'):
        """
        K-Nearest Neighbors classifier.

        Parameters
        ----------
        n_neighbors : int, default=5
            Number of neighbors to use for prediction.

        weights : {'uniform', 'distance'}, default='uniform'
            Weight function used in prediction.
            - 'uniform' : All neighbors have equal weight.
            - 'distance' : Weight points by the inverse of their distance.

        distance_metrics : str, default='euclidean_distances'
            The name of the distance metric to use.
            Supported metrics: 'euclidean_distances', 'manhattan_distances', 'squared_euclidean_distances', 'cosine_distances'.
        """
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.distance_metrics = DistanceMetrics(distance_metrics)

    def fit(self, X, y):
        """
        Fit the model using X as training data and y as target labels.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples, n_features) containing the training data.

        y : np.ndarray
            Array of shape (n_samples,) containing target labels.
        """
        self.n_samples, self.n_features = X.shape
        self.X_train = X
        self.y_train = y

    def get_k_neighbors(self, X):
        """
        Get indices and distances of the k nearest neighbors of each sample in X.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_queries, n_features) containing query samples.

        Returns
        -------
        tuple
            Tuple containing two np.ndarray:
            - Array of shape (n_queries, n_neighbors) with indices of nearest neighbors.
            - Array of shape (n_queries, n_neighbors) with distances to nearest neighbors.
        """
        distances = self.distance_metrics(self.X_train, X)
        indices = np.arange(self.n_samples)
        for i in range(self.n_neighbors):
            for j in range(0, self.n_samples - i - 1):
                if distances[j] < distances[j + 1]:
                    distances[j], distances[j + 1] = distances[j + 1], distances[j]
                    indices[j], indices[j + 1] = indices[j + 1], indices[j]
        Y = self.y_train[indices[-self.n_neighbors:]]
        return Y, distances[-self.n_neighbors:]

    def predict_single(self, X):
        """
        Predict the class label for a single sample.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_features,) containing a single sample.

        Returns
        -------
        int
            Predicted class label.
        """
        neighbors, distances = self.get_k_neighbors(X)
        if self.weights == 'uniform':
            return np.bincount(neighbors).argmax()
        elif self.weights == 'distance':
            distances += 1e-8  # Avoid division by zero
            return np.bincount(neighbors, weights=1 / distances).argmax()
        else:
            raise ValueError(f"Invalid weight function: {self.weights}")

    def predict(self, X):
        """
        Predict class labels for multiple samples.

        Parameters
        ----------
        X : np.ndarray
            Array of shape (n_samples, n_features) containing multiple samples.

        Returns
        -------
        np.ndarray
            Array of shape (n_samples,) containing predicted class labels for each sample in X.
        """
        return np.array([self.predict_single(x) for x in X])
