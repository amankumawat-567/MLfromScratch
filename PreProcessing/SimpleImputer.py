import numpy as np
from Utils import nanmean, nanmedian, nanmode

class SimpleImputer:
    """
    Impute missing values using either 'mean', 'median', 'most_frequent', or 'constant' strategy.

    Parameters
    ----------
    missing_values : int, float, str, default=np.nan
        The placeholder for the missing values.
    strategy : str, default='mean'
        The imputation strategy:
        - 'mean': Replace missing values using the mean along each column.
        - 'median': Replace missing values using the median along each column.
        - 'most_frequent': Replace missing values using the mode along each column.
        - 'constant': Replace missing values with a constant fill_value.
    fill_value : int, float, str, or None, default=None
        When strategy='constant', fill_value is used to replace missing values.

    Attributes
    ----------
    missing_values : int, float, str
        The placeholder for the missing values.
    strategy : str
        The imputation strategy.
    fill_value : int, float, str, or None
        The value used for imputation when strategy='constant'.
    n_features_in_ : int
        Number of features in the input data during fitting.

    Methods
    -------
    fit(X)
        Fit the imputer on input data X.
    transform(X)
        Transform input data X by imputing missing values.
    fit_transform(X)
        Fit the imputer on input data X and transform X in one step.
    """
    def __init__(self, missing_values=np.nan, strategy='mean', fill_value=None):
        self.missing_values = missing_values
        self.strategy = strategy
        self.fill_value = fill_value

    def fit(self, X):
        """
        Fit the imputer on input data X.

        Parameters:
        - X: Input data array with missing values.

        Computes and stores the fill value based on the specified strategy.
        """
        self.n_features_in_ = X.shape[1]

        if self.strategy == 'mean':
            self.fill_value = nanmean(X, axis=0, where=(X != self.missing_values))
        elif self.strategy == 'median':
            self.fill_value = nanmedian(X, axis=0, where=(X != self.missing_values))
        elif self.strategy == 'most_frequent':
            self.fill_value = nanmode(X, axis=0, where=(X != self.missing_values))
        elif self.strategy == 'constant':
            self.fill_value = np.array([self.fill_value]*self.n_features_in_)
        else:
            raise ValueError("Invalid strategy. Must be 'mean', 'median', 'most_frequent', or 'constant'.")

    def transform(self, X):
        """
        Transform input data X by imputing missing values.

        Parameters:
        - X: Input data array with missing values.

        Returns:
        - X_imputed: Transformed array with missing values replaced by fill_value.
        """
        for i in range(self.n_features_in_):
            for j in range(X.shape[0]):
                if X[j, i] == self.missing_values:
                    X[j, i] = self.fill_value[i]
                elif np.isnan(X[j, i]):
                    X[j, i] = self.fill_value[i]
        return X

    def fit_transform(self, X):
        """
        Fit the imputer on input data X and transform X in one step.

        Parameters:
        - X: Input data array with missing values.

        Returns:
        - X_imputed: Transformed array with missing values replaced by fill_value.
        """
        self.fit(X)
        return self.transform(X)