import numpy as np

class GaussianNB:
    """
    Gaussian Naive Bayes (GaussianNB) classifier.

    Parameters
    ----------
    priors : np.ndarray, default=None
        Prior probabilities of the classes. If specified, the priors are not adjusted according to the data.
    var_smoothing : float, default=1e-09
        Portion of the largest variance of all features that is added to variances for calculation stability.

    Methods
    -------
    fit(X, y)
        Fit the Gaussian Naive Bayes model according to the given training data.
    predict(X)
        Perform classification on an array of test vectors X.
    """

    def __init__(self, priors: np.ndarray = None, var_smoothing: float = 1e-09):
        self.priors = priors
        self.var_smoothing = var_smoothing

    def get_likelihood(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of the data given the model parameters.

        Parameters
        ----------
        x : np.ndarray
            Input data sample.

        Returns
        -------
        likelihood : np.ndarray
            Likelihood of each class given the input sample.
        """
        likelihood = np.ones(self.classes)
        for i in range(self.classes):
            for j in range(self.n_features):
                var = self.var[i][j]
                mean = self.mean[i][j]
                # Gaussian likelihood formula
                likelihood[i] *= (1 / np.sqrt(2 * np.pi * var)) * np.exp(-(x[j] - mean) ** 2 / (2 * var))
        return likelihood

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNB':
        """
        Fit the Gaussian Naive Bayes model according to the given training data.

        Parameters
        ----------
        X : np.ndarray
            Training data of shape (n_samples, n_features).
        y : np.ndarray
            Target values of shape (n_samples,).

        Returns
        -------
        self : object
            Fitted estimator.
        """
        n_samples, self.n_features = X.shape
        priors = np.bincount(y)
        self.classes = len(priors)

        self.mean = np.zeros((self.classes, self.n_features))
        self.var = np.zeros((self.classes, self.n_features))

        for i in range(n_samples):
            self.mean[y[i]] += X[i]
            self.var[y[i]] += X[i] ** 2

        for i in range(self.classes):
            self.mean[i] /= priors[i]
            self.var[i] /= priors[i]

        # Calculate variance and apply smoothing
        self.var -= self.mean ** 2

        if self.priors is None:
            self.priors = priors / n_samples

        # Add var_smoothing to variances
        for i in range(self.classes):
            epsilon = self.var_smoothing * np.max(self.var[i])
            self.var[i] += epsilon

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features).

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels for each data sample.
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            likelihood = self.get_likelihood(X[i])
            posterior = likelihood * self.priors
            y_pred[i] = np.argmax(posterior)
        return y_pred