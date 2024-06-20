import numpy as np

class Normalizer:
    """
    Normalize data using either 'minmax' or 'mean' strategy.

    Parameters
    ----------
    strategy : str, default='minmax'
        Strategy for normalization. Must be one of {'minmax', 'mean'}.

    Attributes
    ----------
    strategy : str
        The normalization strategy.
    min_ : np.ndarray or None
        Minimum values of each feature (for 'minmax' strategy).
    max_ : np.ndarray or None
        Maximum values of each feature (for 'minmax' strategy).
    mean_ : np.ndarray or None
        Mean values of each feature (for 'mean' strategy).
    range_ : np.ndarray or None
        Range (max - min) values of each feature (for 'mean' strategy).
    """
    def __init__(self, strategy='minmax'):
        if strategy not in ['minmax', 'mean']:
            raise ValueError("Strategy must be either 'minmax' or 'mean'")
        self.strategy = strategy
        self.min_ = None
        self.max_ = None
        self.mean_ = None
        self.range_ = None

    def fit(self, data):
        """
        Fit the Normalizer to the data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be normalized.

        Raises
        ------
        ValueError
            If the strategy is not recognized.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if self.strategy == 'minmax':
            self.min_ = np.min(data, axis=0)
            self.max_ = np.max(data, axis=0)
        elif self.strategy == 'mean':
            self.mean_ = np.mean(data, axis=0)
            self.range_ = np.ptp(data, axis=0)  # Peak-to-peak (range)

    def transform(self, data):
        """
        Transform the data using the fitted Normalizer.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be normalized.

        Returns
        -------
        np.ndarray
            The normalized data.

        Raises
        ------
        ValueError
            If the Normalizer has not been fitted yet.
        """
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        if self.strategy == 'minmax':
            if self.min_ is None or self.max_ is None:
                raise ValueError("The Normalizer has not been fitted yet.")
            return (data - self.min_) / (self.max_ - self.min_)

        elif self.strategy == 'mean':
            if self.mean_ is None or self.range_ is None:
                raise ValueError("The Normalizer has not been fitted yet.")
            return (data - self.mean_) / self.range_

    def fit_transform(self, data):
        """
        Fit the Normalizer to the data and transform the data.

        Parameters
        ----------
        data : array-like of shape (n_samples, n_features)
            The data to be normalized.

        Returns
        -------
        np.ndarray
            The normalized data.
        """
        self.fit(data)
        return self.transform(data)
