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

class SVR:
    """
    Support Vector Regression (SVR) implementation using Sequential Minimal Optimization (SMO).

    Parameters
    ----------
    kernel : str, default='linear'
        Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', or 'sigmoid'.
    C : float, default=1.0
        Regularization parameter.
    epsilon : float, default=0.01
        Epsilon parameter in the epsilon-insensitive loss function.
    gamma : {'auto', 'scale'} or float, default='auto'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    r : float, default=0
        Coefficient for sigmoid/poly kernels.
    degree : int, default=1
        Degree for poly kernels.
    strategy : {'random', 'max_step', 'max_error'}, default='random'
        Strategy for choosing the second variable in optimization.
    tol : float, default=1e-6
        Tolerance for stopping criteria.
    callback : callable, optional
        A callable that is called after each iteration with the current iteration number and loss value.

    Methods
    -------
    fit(self,X,y,maxIter=1000,logging = False)
        Fit the model according to the given training data.
    predict(X)
        Predict using the SVR model.
    """
    def __init__(self, kernel='linear', C=1.0, epsilon=0.01, gamma="auto", r=0, degree=1, strategy='random', tol=1e-6):
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.Tol = tol
        self.r = r
        self.gamma = gamma
        self.degree = degree
        self.strategy = strategy

    def Kernal_function(self,X1, X2):
        """
        Compute the kernel between X1 and X2.

        Parameters
        ----------
        X1 : ndarray of shape (n_samples_1, n_features)
            First input data.
        X2 : ndarray of shape (n_samples_2, n_features)
            Second input data.

        Returns
        -------
        K : ndarray of shape (n_samples_1, n_samples_2)
            Kernel matrix.
        """
        if self.kernel == 'linear':
            K = np.dot(X1,X2.T)

        elif self.kernel == 'sigmoid':
            K = np.tanh(self.gamma * np.dot(X1,X2.T) + self.r)

        elif self.kernel == 'poly':
            K = (self.gamma * np.dot(X1,X2.T) + self.r)**self.degree

        elif self.kernel == 'rbf':
            K = rbf_kernel(X1,X2,gamma=self.gamma)

        else:
            raise ValueError("This kernel is supported at the moment.")
        return np.round(K,10)

    def Check_KKT(self):
        """
        Check the Karush-Kuhn-Tucker (KKT) conditions.

        Returns
        -------
        lambda_violated : list
            List of indices where KKT conditions are violated.
        """
        lambda_violated = []
        for i in range(self.n_samples):
            l = self.lambda_[i]
            e = abs(self.E[i])
            if l == 0 and e < self.epsilon:
                  continue
            elif (l < self.C and l > -self.C) and e == self.epsilon:
                  continue
            elif (l == self.C or l == -self.C) and e > self.epsilon:
                  continue
            lambda_violated.append(i)
        return lambda_violated

    def Heuristics(self,b):
        """
        Select the second variable based on the specified strategy.

        Parameters
        ----------
        b : int
            Index of the first variable.

        Returns
        -------
        a : int
            Index of the second variable.
        """
        if self.strategy == 'random':
            a = np.random.randint(0, self.n_samples)
            while a == b:
                  a = np.random.randint(0, self.n_samples)

        elif self.strategy == 'max_step':
            maxStep = -1
            for i in range(self.n_samples):
                distance = np.sum((self.X[i] - self.X[b])**2)
                if distance > maxStep:
                    a = i

        elif self.strategy == 'max_error':
            maxError = -1
            for i in range(self.n_samples):
                error = abs((self.E[i] - self.E[b]))
                if distance > maxError:
                    a = i
        else:
            raise ValueError("This strategy is supported at the moment.")
        return a

    def predict(self,X):
        """
        Predict using the SVR model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        kerneled_matrix  = self.Kernal_function(self.X,X)
        return np.round(np.dot(self.lambda_,kerneled_matrix) + self.lambda_0,10)

    def log_iteration(self, iteration, loss, log_fn=print):
        """
        Logs the current iteration and loss value using the specified logging function.
        
        Parameters:
            iteration (int): The current iteration number.
            loss (float): The loss value.
            log_fn (function): The logging function to use (default is print).
        """
        log_message = f"Iteration {iteration} -> Loss: {loss:.6f}"
        log_fn(log_message)

    def compute_gradient(self, y,k):
        """
        Compute the gradient of the loss function.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.
        k : ndarray of shape (n_samples, n_samples)
            Kernel matrix.

        Returns
        -------
        gradient : ndarray of shape (n_samples,)
            Gradient of the loss function.
        """
        # Compute the derivatives for L1, L2, and L3
        dL1 = self.epsilon * np.sign(self.lambda_)
        dL2 = -y
        dL3 = np.dot(k, self.lambda_)

        # Sum the derivatives to get the gradient
        gradient = dL1 + dL2 + dL3

        return gradient

    def fit(self,X,y,maxIter=1000,logging = False):
        """
        Fit the model according to the given training data using SMO.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        max_iter : int, default=1000
            Maximum number of iterations.
        logging : bool, default=False
            Whether to log the iterations and loss values.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        self.n_samples, self.n_features = X.shape
        if (self.gamma == "scale"):
            self.gamma = 1.0 / (self.n_features * X.var())
        elif (self.gamma == "auto"):
            self.gamma = 1.0 / self.n_features

        self.lambda_ = np.zeros(self.n_samples)
        self.lambda_0 = 0
        K = self.Kernal_function(self.X,X)

        iter = 0

        f = np.round(np.dot(self.lambda_,K) + self.lambda_0,10)
        self.E = np.round(y-f,10)

        prev_gradient = self.compute_gradient(y,K)

        while iter < maxIter:
            lambda_violated = self.Check_KKT()
            if len(lambda_violated) == 0:
                break

            for b in lambda_violated:
                a = self.Heuristics(b)

                s = np.round(self.lambda_[b]+ self.lambda_[a],10)
                eta = np.round(K[a,a] + K[b,b] - 2*K[a,b],10)

                if eta <=0:
                  continue

                is_lambda_b_updated = False

                for phi in [-2,0,2]:
                    lambda_raw = np.round(self.lambda_[b] + (self.E[b] - self.E[a] + self.epsilon*phi)/eta,10)
                    if np.sign(s - lambda_raw) - np.sign(lambda_raw) == phi:
                        is_lambda_b_updated = True
                        break

                if not is_lambda_b_updated:
                    va = f[a] - self.lambda_[a]*K[a,a] - self.lambda_[b]*K[b,a] - self.lambda_0
                    vb = f[b] - self.lambda_[a]*K[a,b] - self.lambda_[b]*K[b,b] - self.lambda_0
                    for lambda_raw in [0,s]:
                        positive_perturbation = self.epsilon*(np.sign(lambda_raw+0.01) - np.sign(s - (lambda_raw+0.01))) + y[a] - y[b] + ((lambda_raw+0.01)-s)*K[a,a] + (lambda_raw+0.01)*K[b,b] + (s-2*(lambda_raw+0.01))*K[a,b] - va + vb
                        negitive_perturbation = self.epsilon*(np.sign(lambda_raw-0.01) - np.sign(s - (lambda_raw-0.01))) + y[a] - y[b] + ((lambda_raw-0.01)-s)*K[a,a] + (lambda_raw-0.01)*K[b,b] + (s-2*(lambda_raw-0.01))*K[a,b] - va + vb
                        positive_perturbation = np.round(positive_perturbation,10)
                        negitive_perturbation = np.round(negitive_perturbation,10)
                        if (positive_perturbation >=0) and (negitive_perturbation<=0):
                            is_lambda_b_updated = True
                            break

                L = max(s - self.C,-self.C)
                H = min(self.C, s + self.C)
                lambda_raw = np.clip(lambda_raw,L,H)

                old_lambda_a, old_lambda_b = self.lambda_[a],self.lambda_[b]
                self.lambda_[a],self.lambda_[b] = (s-lambda_raw),lambda_raw

                l0a = self.E[a] + (self.lambda_[a] - old_lambda_a)*K[a,a] + (self.lambda_[b] - old_lambda_b)*K[a,b] + self.lambda_0
                l0b = self.E[b] + (self.lambda_[a] - old_lambda_a)*K[b,a] + (self.lambda_[b] - old_lambda_b)*K[b,b] + self.lambda_0

                self.lambda_0 = np.round((l0a + l0b)/2,10)

                f = np.round(np.dot(self.lambda_,K) + self.lambda_0,10)
                self.E = np.round(y-f,10)
                Loss = np.round(np.sum(self.E**2)/self.n_samples,10)

                if Loss < 1e-10:
                    print("Over fitting")
                    iter = maxIter+1
                    break

            curr_gradient = self.compute_gradient(y,K)
            if np.sum(abs(curr_gradient - prev_gradient)) < self.Tol :
                break
            prev_gradient = curr_gradient

            iter += 1
            if logging:
                self.log_iteration(iter,Loss)

        return self
    