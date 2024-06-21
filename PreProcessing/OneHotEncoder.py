import numpy as np

class OneHotEncoder:
    """
    Encode categorical labels into one-hot encoded vectors and vice versa using numpy arrays.

    Attributes:
    -----------
    categories_ : numpy.ndarray
        The unique categories found during the fit operation.
    num_categories_ : int
        The number of unique categories.

    Methods:
    --------
    fit(y):
        Fit encoder to the given list of categorical labels `y`.

    transform(y):
        Transform categorical labels `y` into one-hot encoded vectors.

    fit_transform(y):
        Fit encoder to `y` and transform `y` simultaneously.

    inverse_transform(y):
        Convert one-hot encoded vectors `y` back to original categorical labels.

    Restrictions:
    -------------
    - `y` must be a 1-dimensional numpy array.
    """

    def __init__(self):
        self.categories_ = None
        self.num_categories_ = None
        
    def fit(self, y):
        """
        Fit encoder to the given list of categorical labels `y`.

        Parameters:
        -----------
        y : numpy.ndarray
            The categorical labels to fit the encoder on.
        """
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional numpy array")
        
        self.categories_ = np.unique(y)
        self.num_categories_ = len(self.categories_)
        
    def transform(self, y):
        """
        Transform categorical labels `y` into one-hot encoded vectors.

        Parameters:
        -----------
        y : numpy.ndarray
            The categorical labels to transform.

        Returns:
        --------
        numpy.ndarray
            One-hot encoded vectors corresponding to each input label.
        """
        if self.categories_ is None:
            raise ValueError("fit must be called before transform")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional numpy array")
        
        encoded = np.zeros((len(y), self.num_categories_), dtype=int)
        category_to_index = {cat: idx for idx, cat in enumerate(self.categories_)}
        
        for i, cat in enumerate(y):
            encoded[i, category_to_index[cat]] = 1
        
        return encoded
    
    def fit_transform(self, y):
        """
        Fit encoder to `y` and transform `y` simultaneously.

        Parameters:
        -----------
        y : numpy.ndarray
            The categorical labels to fit and transform.

        Returns:
        --------
        numpy.ndarray
            One-hot encoded vectors corresponding to each input label.
        """
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """
        Convert one-hot encoded vectors `y` back to original categorical labels.

        Parameters:
        -----------
        y : numpy.ndarray
            One-hot encoded vectors to convert back to original labels.

        Returns:
        --------
        numpy.ndarray
            Original categorical labels corresponding to each input vector.
        """
        if self.categories_ is None:
            raise ValueError("fit must be called before inverse_transform")
        
        if y.ndim != 2:
            raise ValueError("y must be a 2-dimensional numpy array")
        
        indices = np.argmax(y, axis=1)
        original = np.array([self.categories_[idx] for idx in indices])
        return original
