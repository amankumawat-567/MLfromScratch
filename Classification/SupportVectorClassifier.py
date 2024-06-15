import numpy as np

def euclidean_distances(X, Y, squared=True):
    """
    Compute the (squared) Euclidean distances between each pair of vectors in X and Y.

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Input data.

    Y : array-like of shape (n_samples_Y, n_features)
        Input data.

    squared : bool, optional, default=True
        Return squared Euclidean distances.

    Returns
    -------
    distances : ndarray of shape (n_samples_X, n_samples_Y)
        Distances between each pair of vectors.

    Raises
    ------
    ValueError
        If X and Y do not have the same number of features.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("X and Y must have the same number of features")

    # Compute the squared Euclidean distance
    XX = np.sum(X ** 2, axis=1)[:, np.newaxis]
    YY = np.sum(Y ** 2, axis=1)[np.newaxis, :]
    distances = XX + YY - 2 * np.dot(X, Y.T)

    # Ensure all distances are non-negative (can be slightly negative due to numerical errors)
    np.maximum(distances, 0, out=distances)

    if not squared:
        # Take the square root of distances while ensuring the array is float64
        np.sqrt(distances, out=distances.astype(np.float64))

    return distances

def rbf_kernel(X, Y, gamma=None):
    """
    Compute the RBF (Gaussian) kernel between X and Y.

    K(x, y) = exp(-gamma ||x - y||^2)

    Parameters
    ----------
    X : array-like of shape (n_samples_X, n_features)
        Input data.

    Y : array-like of shape (n_samples_Y, n_features)
        Input data.

    gamma : float, optional
        If None, defaults to 1.0 / n_features.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        The RBF kernel matrix.
    """
    if gamma is None:
        gamma = 1.0 / X.shape[1]

    # Compute the squared Euclidean distances
    K = euclidean_distances(X, Y, squared=True)

    # Apply the RBF kernel transformation
    K *= -gamma
    np.exp(K, out=K)  # Exponentiate K in-place

    return K

class SVC:
    def __init__(self, kernel='linear', epsilon=1e-9, C=1.0, gamma="auto", r=0, degree=1, strategy='random', tol=1e-6):
        """
        Initialize the Support Vector Classifier (SVC) model.

        Parameters
        ----------
        kernel : str, default='linear'
            Specifies the kernel type to be used in the algorithm ('linear', 'sigmoid', 'poly', 'rbf').
        epsilon : float, default=1e-9
            Small value to avoid numerical instability in logarithm computation.
        C : float, default=1.0
            Regularization parameter.
        gamma : float or {'auto', 'scale'}, default='auto'
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If 'auto', uses 1 / n_features.
            If 'scale', uses 1 / (n_features * X.var()).
        r : float, default=0
            Parameter for 'sigmoid' and 'poly' kernels.
        degree : int, default=1
            Degree for 'poly' kernel.
        strategy : str, default='random'
            Strategy to select the second alpha (a) in SMO ('random' or 'max_step').
        tol : float, default=1e-6
            Tolerance for stopping criterion.

        Attributes
        ----------
        C : float
            Regularization parameter.
        kernel : str
            Kernel function to be used.
        tol : float
            Tolerance for stopping criterion.
        r : float
            Parameter for 'sigmoid' and 'poly' kernels.
        gamma : float
            Kernel coefficient.
        degree : int
            Degree for 'poly' kernel.
        strategy : str
            Strategy to select the second alpha in SMO.
        epsilon : float
            Small value to avoid numerical instability in logarithm computation.
        X : ndarray or None
            Training feature array. Set during fitting.
        y : ndarray or None
            Training labels (-1 or 1). Set during fitting.
        alpha : ndarray or None
            Lagrange multipliers. Set during fitting.
        alpha_0 : float
            Intercept term in decision function.
        """
        self.C = C
        self.kernel = kernel
        self.tol = tol
        self.r = r
        self.gamma = gamma
        self.degree = degree
        self.strategy = strategy
        self.epsilon = epsilon
        self.X = None
        self.y = None
        self.alpha = None
        self.alpha_0 = 0

    def kernel_function(self, X1, X2):
        """
        Compute the kernel function based on the selected kernel type.

        Parameters
        ----------
        X1 : ndarray
            First feature array.
        X2 : ndarray
            Second feature array.

        Returns
        -------
        K : ndarray
            Kernel matrix computed based on the selected kernel type.
        """
        if self.kernel == 'linear':
            K = np.dot(X1, X2.T)
        elif self.kernel == 'sigmoid':
            K = np.tanh(self.gamma * np.dot(X1, X2.T) + self.r)
        elif self.kernel == 'poly':
            K = (self.gamma * np.dot(X1, X2.T) + self.r) ** self.degree
        elif self.kernel == 'rbf':
            K = rbf_kernel(X1, X2, gamma=self.gamma)
        else:
            raise ValueError("Unsupported kernel type. Choose from 'linear', 'sigmoid', 'poly', or 'rbf'.")
        return np.round(K, 10)

    def check_kkt(self):
        """
        Check for KKT violations and return indices of violated alphas.

        Returns
        -------
        alpha_violated : list
            List of indices with violated KKT conditions.
        """
        alpha_violated = []
        f = self.predict(self.X)
        for i in range(self.n_samples):
            a = self.alpha[i]
            e = self.y[i] * f[i]
            if a == 0 and e >= 1:
                continue
            elif 0 < a < self.C and e == 1:
                continue
            elif a == self.C and e <= 1:
                continue
            alpha_violated.append(i)
        return alpha_violated

    def heuristics(self, b):
        """
        Determine index 'a' based on the selected strategy.

        Parameters
        ----------
        b : int
            Index to exclude when selecting 'a'.

        Returns
        -------
        a : int
            Selected index 'a' based on the strategy.
        """
        if self.strategy == 'random':
            a = np.random.randint(0, self.n_samples)
            while a == b:
                a = np.random.randint(0, self.n_samples)
        elif self.strategy == 'max_step':
            max_step = -1
            a = 0  # Initialize 'a'
            for i in range(self.n_samples):
                distance = np.sum((self.X[i] - self.X[b]) ** 2)
                if distance > max_step:
                    max_step = distance
                    a = i
        else:
            raise ValueError("Unsupported strategy. Choose from 'random' or 'max_step'.")
        return a

    def binary_cross_entropy(self, y_true, y_pred):
        """
        Compute binary cross-entropy loss for a single prediction.

        Parameters
        ----------
        y_true : int
            True class label (0 or 1).
        y_pred : float
            Predicted probability of class 1.

        Returns
        -------
        loss : float
            Binary cross-entropy loss.
        """
        if y_true == 1:
            return -np.log(y_pred)
        else:
            return -np.log(1 - y_pred)

    def loss_function(self, y_true, y_pred):
        """
        Compute average binary cross-entropy loss over all samples.

        Parameters
        ----------
        y_true : ndarray
            True class labels (0 or 1) for all samples.
        y_pred : ndarray
            Predicted probabilities of class 1 for all samples.

        Returns
        -------
        loss : float
            Average binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        N = y_true.shape[0]
        total_loss = np.sum([self.binary_cross_entropy(y_true[i], y_pred[i]) for i in range(N)])
        return total_loss / N

    def predict(self, X):
        """
        Predict class labels for input data X.

        Parameters
        ----------
        X : ndarray
            Input feature array.

        Returns
        -------
        prediction : ndarray
            Predicted class labels (0 or 1) for each sample in X.
        """
        kerneled_matrix = self.kernel_function(self.X, X)
        prediction = np.dot(self.alpha * self.y, kerneled_matrix) + self.alpha_0
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = 0
        return prediction

    def log_iteration(self, iteration, y_true, y_pred, log_fn=print):
        """
        Log iteration details including loss.

        Parameters
        ----------
        iteration : int
            Current iteration number.
        y_true : ndarray
            True class labels (0 or 1) for all samples.
        y_pred : ndarray
            Predicted probabilities of class 1 for all samples.
        log_fn : function, optional
            Logging function. Default is print.
        """
        y_pred[y_pred == -1] = 0
        loss = self.loss_function(y_true, y_pred)
        log_message = f"Iteration {iteration} -> Loss: {loss:.6f}"
        log_fn(log_message)

    def compute_gradient(self, K):
        """
        Compute gradient of the objective function w.r.t. alpha.

        Parameters
        ----------
        K : ndarray
            Kernel matrix.

        Returns
        -------
        gradient : ndarray
            Gradient of the objective function w.r.t. alpha.
        """
        gradient = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            gradient[i] = np.sum(self.alpha * self.y * self.y[i] * K[i, :]) - 1
        return gradient

    def fit(self, X, y, max_iter=1000, logging=False):
        """
        Fit the SVC model using Sequential Minimal Optimization (SMO) algorithm.

        Parameters
        ----------
        X : ndarray
            Input feature array.
        y : ndarray
            True class labels (0 or 1) for all samples.
        max_iter : int, optional
            Maximum number of iterations. Default is 1000.
        logging : bool, optional
            Whether to log progress during training. Default is False.

        Returns
        -------
        self : object
            Returns self.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        self.y = y
        self.y[y == 0] = -1

        if self.gamma == "scale":
            self.gamma = 1.0 / (self.n_features * X.var())
        elif self.gamma == "auto":
            self.gamma = 1.0 / self.n_features

        self.alpha = np.zeros(self.n_samples)
        self.alpha_0 = 0
        K = self.kernel_function(self.X, X)

        iter = 0
        prev_gradient = self.compute_gradient(K)

        while iter < max_iter:
            f = np.sign(np.dot(self.alpha * self.y, K) + self.alpha_0)

            alpha_violated = self.check_kkt()
            if len(alpha_violated) == 0:
                break

            for b in alpha_violated:
                a = self.heuristics(b)

                s = np.round(self.alpha[b] * self.y[b] * self.y[a] + self.alpha[a], 10)
                eta = np.round(K[a, a] + K[b, b] - 2 * K[a, b], 10)

                if eta <= 0:
                    continue

                alpha_raw = np.round(self.alpha[b] + self.y[b] * ((self.y[b] - f[b]) - (self.y[a] - f[a])) / eta, 10)

                if self.y[a] == self.y[b]:
                    L = max(0, self.alpha[b] + self.alpha[a] - self.C)
                    H = min(self.C, self.alpha[b] + self.alpha[a])
                else:
                    L = max(0, self.alpha[b] - self.alpha[a])
                    H = min(self.C, self.C + self.alpha[b] - self.alpha[a])

                old_alpha_a, old_alpha_b = self.alpha[a], self.alpha[b]

                self.alpha[b] = np.round(np.clip(alpha_raw, L, H), 10)

                self.alpha[a] = np.round(s - self.alpha[b] * self.y[b] * self.y[a], 10)

                alpha_0_a = self.y[a] - f[a] + (self.alpha[a] - old_alpha_a) * self.y[a] * K[a, a] + (
                            self.alpha[b] - old_alpha_b) * self.y[b] * K[a, b] + self.alpha_0
                alpha_0_b = self.y[b] - f[b] + (self.alpha[a] - old_alpha_a) * self.y[a] * K[b, a] + (
                            self.alpha[b] - old_alpha_b) * self.y[b] * K[b, b] + self.alpha_0

                self.alpha_0 = np.round((alpha_0_a + alpha_0_b) / 2, 10)

            curr_gradient = self.compute_gradient(K)
            if np.sum(abs(curr_gradient - prev_gradient)) < self.tol:
                break
            prev_gradient = curr_gradient

            iter += 1
            if logging:
                self.log_iteration(iter, y, f)

        return self

