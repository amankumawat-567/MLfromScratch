import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.01, max_iter: int = 100, logging: bool = False, epsilon: float = 1e-9):
        """
        Logistic Regression classifier.

        Parameters
        ----------
        learning_rate : float, default=0.01
            The learning rate for gradient descent.
        
        max_iter : int, default=100
            The maximum number of iterations for gradient descent.
        
        logging : bool, default=False
            Whether to log the loss at each iteration.
        
        epsilon : float, default=1e-9
            A small value to avoid log(0) in the loss function.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.logging = logging
        self.epsilon = epsilon

    def log_iteration(self, iteration: int, y_true: np.ndarray, y_pred: np.ndarray, log_fn=print) -> None:
        """
        Log the loss at each iteration.

        Parameters
        ----------
        iteration : int
            The current iteration number.
        
        y_true : np.ndarray
            The true labels.
        
        y_pred : np.ndarray
            The predicted labels.
        
        log_fn : function, default=print
            The logging function to use.
        """
        loss = self.loss_function(y_true, y_pred)
        log_message = f"Iteration {iteration} -> Loss: {loss:.6f}"
        log_fn(log_message)

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Apply the sigmoid function.

        Parameters
        ----------
        z : np.ndarray
            The input values.

        Returns
        -------
        np.ndarray
            The sigmoid of the input values.
        """
        return 1 / (1 + np.exp(-z))

    def binary_cross_entropy(self, y_true: float, y_pred: float) -> float:
        """
        Compute the binary cross-entropy loss for a single sample.

        Parameters
        ----------
        y_true : float
            The true label.
        
        y_pred : float
            The predicted label.

        Returns
        -------
        float
            The binary cross-entropy loss.
        """
        if y_true == 1:
            return -np.log(y_pred)
        else:
            return -np.log(1 - y_pred)

    def feed_forward(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the output of the model.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        np.ndarray
            The predicted probabilities.
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def loss_function(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute the loss function.

        Parameters
        ----------
        y_true : np.ndarray
            The true labels.
        
        y_pred : np.ndarray
            The predicted labels.

        Returns
        -------
        float
            The average binary cross-entropy loss.
        """
        y_pred = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        N = y_true.shape[0]
        total_loss = 0
        for i in range(N):
            total_loss += self.binary_cross_entropy(y_true[i], y_pred[i])
        return total_loss / N

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the logistic regression model.

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features).
        
        y : np.ndarray
            The target values of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for iter in range(self.max_iter):
            y_pred = self.feed_forward(X)
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if self.logging:
                self.log_iteration(iter + 1, y, y_pred)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the class labels for the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            The predicted class labels.
        """
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return np.round(y_pred)
