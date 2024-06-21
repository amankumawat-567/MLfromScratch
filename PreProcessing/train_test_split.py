import numpy as np

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the data into random train and test subsets.

    Parameters:
    -----------
    X : numpy.ndarray
        The input data array or matrix.

    y : numpy.ndarray
        The target labels.

    test_size : float, optional (default=0.2)
        The proportion of the dataset to include in the test split.
        Should be between 0.0 and 1.0.

    random_state : int, optional (default=None)
        Controls the random seed for reproducibility.

    Returns:
    --------
    X_train : numpy.ndarray
        The training input data.

    X_test : numpy.ndarray
        The testing input data.

    y_train : numpy.ndarray
        The training target labels.

    y_test : numpy.ndarray
        The testing target labels.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    
    if not (0.0 <= test_size <= 1.0):
        raise ValueError("test_size should be between 0.0 and 1.0")
    
    # Set random seed if specified
    if random_state is not None:
        np.random.seed(random_state)
    
    # Number of samples
    n_samples = X.shape[0]
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    
    # Calculate number of samples in test set
    test_samples = int(n_samples * test_size)
    
    # Split indices into train and test
    test_indices = indices[:test_samples]
    train_indices = indices[test_samples:]
    
    # Split data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test
