import numpy as np

class LinearRegression:
    """
    A simple implementation of Linear Regression.

    Methods
    -------
    fit(X, Y, method="matrix")
        Fit the linear model to the data.
    predict(X)
        Predict using the linear model.
    """
    
    def __init__(self):
        """Initialize the Linear Regression model."""
        self.__weights = None

    def __matrix_fit(self, X, Y):
        """
        Fit the linear model using the normal equation (matrix method).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples,)
            Target values.

        Raises
        ------
        Exception
            If the matrix is singular (multicollinearity in the data).
        """
        intercept_column = np.ones((X.shape[0], 1))
        X = np.hstack((intercept_column, X))
        x_xT = np.dot(X.T, X)
        det = np.linalg.det(x_xT)
        if det == 0:
            raise Exception("Multicollinearity in the data")

        x_xT_inv = np.linalg.inv(x_xT)
        self.__weights = np.round(np.dot(x_xT_inv, np.dot(X.T, Y)), 10)

    def __stats_fit(self, X, Y):
        """
        Fit the linear model using statistical method.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples,)
            Target values.
        """
        n_samples, n_features = X.shape
        x_mean = np.mean(X, axis=0)
        y_mean = np.mean(Y)
        self.__weights = np.dot((Y - y_mean), (X - x_mean)) / np.sum((X - x_mean) ** 2, axis=0)
        bias = y_mean - np.dot(x_mean, self.__weights)
        self.__weights = np.insert(self.__weights, 0, bias)

    def fit(self, X, Y, method="matrix"):
        """
        Fit the linear model to the data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        Y : ndarray of shape (n_samples,)
            Target values.
        method : {"matrix", "stats"}, default="matrix"
            Method to use for fitting the model.
        
        Raises
        ------
        Exception
            If an invalid method is provided.
        """
        if method == "stats":
            self.__stats_fit(X, Y)
        elif method == "matrix":
            self.__matrix_fit(X, Y)
        else:
            raise Exception("Invalid method")

    def predict(self, X):
        """
        Predict using the linear model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        Pred : ndarray of shape (n_samples,)
            Returns predicted values.
        """
        intercept_column = np.ones((X.shape[0], 1))
        X = np.hstack((intercept_column, X))
        Pred = np.dot(X, self.__weights)
        return Pred