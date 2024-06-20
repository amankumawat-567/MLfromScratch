import numpy as np

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.

    Attributes
    ----------
    mean_ : np.ndarray or None
        The mean value for each feature during fitting.
    std_ : np.ndarray or None
        The standard deviation for each feature during fitting.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        """
        Compute the mean and standard deviation for normalization.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be standardized.

        Returns
        -------
        self : object
            Fitted StandardScaler object.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)

        return self

    def transform(self, data):
        """
        Standardize the data using the mean and standard deviation.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be standardized.

        Returns
        -------
        np.ndarray
            The standardized data.

        Raises
        ------
        ValueError
            If the StandardScaler has not been fitted yet.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if self.mean_ is None or self.std_ is None:
            raise ValueError("The StandardScaler has not been fitted yet.")

        return (data - self.mean_) / self.std_

    def fit_transform(self, data):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be standardized.

        Returns
        -------
        np.ndarray
            The standardized data.
        """
        self.fit(data)
        return self.transform(data)
