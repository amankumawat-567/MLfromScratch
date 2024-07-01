import numpy as np

class AgglomerativeClustering:
    """
    A class to perform Agglomerative Hierarchical Clustering.

    Parameters
    ----------
    n_clusters : int, optional, default=2
        The number of clusters to form.
    linkage : str, optional, default='single'
        The linkage criterion to use. Options are 'single', 'complete', 'average', and 'ward'.
    metric : str, optional, default='euclidean'
        The distance metric to use. Options are 'euclidean', 'manhattan', 'cosine', 'l1', and 'l2'.
    """
    
    def __init__(self, n_clusters=2, linkage='single', metric='euclidean'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric

    def matric(self, X, Y):
        """
        Compute the distance between two points X and Y based on the specified metric.

        Parameters
        ----------
        X : array-like
            First point.
        Y : array-like
            Second point.

        Returns
        -------
        float
            The computed distance between X and Y.
        """
        if self.metric == 'euclidean':
            return np.linalg.norm(X - Y)
        elif self.metric == 'manhattan':
            return np.sum(np.abs(X - Y))
        elif self.metric == 'cosine':
            return 1 - np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))
        elif self.metric == 'l1':
            return np.sum(np.abs(X - Y))
        elif self.metric == 'l2':
            return np.linalg.norm(X - Y)
        else:
            raise ValueError("Unknown metric type: %s" % self.metric)

    def make_distance_matrix(self, n_samples, X):
        """
        Create a distance matrix for the data points.

        Parameters
        ----------
        n_samples : int
            Number of samples in the dataset.
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        distance_matrix : array, shape (n_samples, n_samples)
            The distance matrix for the data points.
        """
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distance_matrix[i][j] = self.matric(X[i], X[j])
                distance_matrix[j][i] = distance_matrix[i][j]
        np.fill_diagonal(distance_matrix, np.inf)
        return distance_matrix

    def ArgMIN(self, distance_matrix):
        """
        Find the indices of the minimum value in the distance matrix.

        Parameters
        ----------
        distance_matrix : array, shape (n_samples, n_samples)
            The distance matrix for the data points.

        Returns
        -------
        min_index : tuple
            The indices of the minimum value in the distance matrix.
        """
        min_value = np.inf
        min_index = None
        for i in range(len(distance_matrix)):
            for j in range(i+1, len(distance_matrix)):
                if distance_matrix[i][j] < min_value:
                    min_value = distance_matrix[i][j]
                    min_index = (i, j)
        return min_index

    def compute_linkage(self, c1, c2, X):
        """
        Compute the linkage distance between two clusters.

        Parameters
        ----------
        c1 : list
            Indices of points in the first cluster.
        c2 : list
            Indices of points in the second cluster.
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        float
            The computed linkage distance between the two clusters.
        """
        if self.linkage == 'single':
            return np.min([self.matric(X[i], X[j]) for i in c1 for j in c2])
        elif self.linkage == 'complete':
            return np.max([self.matric(X[i], X[j]) for i in c1 for j in c2])
        elif self.linkage == 'average':
            return np.mean([self.matric(X[i], X[j]) for i in c1 for j in c2])
        elif self.linkage == 'ward':
            return np.sum((X[c1].mean(axis=0) - X[c2].mean(axis=0)) ** 2)
        else:
            raise ValueError("Unknown linkage type: %s" % self.linkage)

    def fit(self, X):
        """
        Perform hierarchical/agglomerative clustering on the data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        n_samples = len(X)
        distance_matrix = self.make_distance_matrix(n_samples, X)
        clusters = []
        for i in range(n_samples):
            clusters.append([i])

        while len(clusters) > self.n_clusters:
            min_index = self.ArgMIN(distance_matrix)
            i, j = min_index
            clusters[j] += clusters[i]
            del clusters[i]
            distance_matrix = np.delete(np.delete(distance_matrix, i, 0), i, 1)
            proximity_array = np.zeros(len(clusters))
            for i in range(len(clusters)):
                proximity_array[i] = self.compute_linkage(clusters[i], clusters[j-1], X)
            distance_matrix[:, j-1] = proximity_array
            distance_matrix[j-1] = proximity_array
        
        self.clusters = clusters
        self.distance_matrix = distance_matrix
        return self

    def fit_predict(self, X):
        """
        Perform clustering on the data and return cluster labels.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data.

        Returns
        -------
        labels : array, shape (n_samples,)
            Cluster labels for each point in the dataset.
        """
        self.fit(X)
        labels = np.zeros(len(X)).astype(int)
        for i, cluster in enumerate(self.clusters):
            for sample in cluster:
                labels[sample] = i
        return labels
