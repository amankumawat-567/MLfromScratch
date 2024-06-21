import numpy as np

class LabelEncoder:
    """
    Encode categorical labels into numerical indices and vice versa using numpy arrays.

    Attributes:
    -----------
    classes_ : numpy.ndarray
        The unique classes found during the fit operation.

    Methods:
    --------
    fit(y):
        Fit label encoder to the given list of labels `y`.

    transform(y):
        Transform labels `y` into numerical indices based on the fitted encoder.

    fit_transform(y):
        Fit label encoder to `y` and transform `y` simultaneously.

    inverse_transform(y):
        Convert numerical indices `y` back to original labels.

    Restrictions:
    -------------
    - `y` must be a 1-dimensional numpy array.
    """

    def __init__(self):
        self.classes_ = None
        
    def fit(self, y):
        """
        Fit label encoder to the given list of labels `y`.

        Parameters:
        -----------
        y : numpy.ndarray
            The categorical labels to fit the encoder on.
        """
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional numpy array")
        
        self.classes_ = np.unique(y)
        
    def transform(self, y):
        """
        Transform labels `y` into numerical indices based on the fitted encoder.

        Parameters:
        -----------
        y : numpy.ndarray
            The categorical labels to transform.

        Returns:
        --------
        numpy.ndarray
            Transformed numerical indices corresponding to each input label.
        """
        if self.classes_ is None:
            raise ValueError("fit must be called before transform")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional numpy array")
        
        class_to_index = {cl: idx for idx, cl in enumerate(self.classes_)}
        transformed = np.array([class_to_index[cl] for cl in y])
        return transformed
    
    def fit_transform(self, y):
        """
        Fit label encoder to `y` and transform `y` simultaneously.

        Parameters:
        -----------
        y : numpy.ndarray
            The categorical labels to fit and transform.

        Returns:
        --------
        numpy.ndarray
            Transformed numerical indices corresponding to each input label.
        """
        self.fit(y)
        return self.transform(y)
    
    def inverse_transform(self, y):
        """
        Convert numerical indices `y` back to original labels.

        Parameters:
        -----------
        y : numpy.ndarray
            Numerical indices to convert back to original labels.

        Returns:
        --------
        numpy.ndarray
            Original categorical labels corresponding to each input index.
        """
        if self.classes_ is None:
            raise ValueError("fit must be called before inverse_transform")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1-dimensional numpy array")
        
        index_to_class = {idx: cl for idx, cl in enumerate(self.classes_)}
        original = np.array([index_to_class[idx] for idx in y])
        return original
