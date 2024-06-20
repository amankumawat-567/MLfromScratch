import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier, Node

class RandomForestClassifier(DecisionTreeClassifier):
    """
    Random Forest Classifier implementation.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of decision trees in the forest.
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node in each tree.
    max_depth : int or None, optional
        Maximum depth of each decision tree. If None, nodes are expanded until all leaves are pure or until min_samples_split is reached.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split in each tree.
    max_features : {'auto', 'sqrt', 'log2'} or int, default='auto'
        Number of features to consider when looking for the best split. If 'auto', max_features=sqrt(n_features).
    bootstrap_fraction : float or None, default=None
        Fraction of samples to be used for bootstrapping. If None, all samples are used.

    Attributes
    ----------
    n_estimators : int
        Number of decision trees in the forest.
    max_features : int
        Number of features to consider when looking for the best split.
    bootstrap_fraction : float or None
        Fraction of samples to be used for bootstrapping.
    trees : np.ndarray
        Array of decision trees in the forest.
    """
    def __init__(self, n_estimators=100, min_samples_split=2, max_depth=None, criterion='gini', max_features='auto', bootstrap_fraction=None):
        super().__init__(min_samples_split=min_samples_split, max_depth=max_depth, criterion=criterion)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap_fraction = bootstrap_fraction
        self.trees = np.empty(n_estimators, dtype=object)

    def bootstrap_resample(self, dataset):
        """
        Perform bootstrapping to create a resampled dataset.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset with features and labels.

        Returns
        -------
        np.ndarray
            Bootstrapped sample from the dataset.
        """
        if self.bootstrap_fraction is None:
            selected_rows = np.random.randint(0, self.num_samples, self.num_samples)
        else:
            num_to_change = int(self.bootstrap_fraction * self.num_samples)
            selected_rows = np.arange(self.num_samples)
            index_to_change = np.random.choice(self.num_samples, num_to_change, replace=False)
            new_values = np.random.randint(0, self.num_samples, num_to_change)
            selected_rows[index_to_change] = new_values

        selected_columns = np.random.choice(self.features, self.max_features, replace=False)
        selected_columns = np.append(selected_columns, -1)

        bootstrapped_sample = dataset[:, selected_columns][selected_rows]
        return bootstrapped_sample

    def fit(self, X, y):
        """
        Fit the Random Forest classifier to the training data.

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
        dataset = np.column_stack((X, y))
        self.num_samples, self.features = X.shape
        self.num_labels = len(np.unique(y))

        # Determine max_features based on the specified option
        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(self.features))
        elif self.max_features == 'auto':
            self.max_features = self.features

        self.n_features = self.max_features

        # Set default max_depth if not specified
        if self.max_depth is None:
            self.max_depth = self.num_samples

        # Build each decision tree in the forest
        for i in range(self.n_estimators):
            bootstrapped_sample = self.bootstrap_resample(dataset)
            num_samples_per_class = np.bincount(bootstrapped_sample[:, -1].astype(int))
            self.trees[i] = self.build_tree(bootstrapped_sample, num_samples_per_class)

        return self

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : np.ndarray
            Test data of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predicted class labels for each sample in X.
        """
        num_samples = X.shape[0]
        predictions = np.empty(num_samples)
        for j in range(num_samples):
            predictions_per_class = np.zeros(self.num_labels)
            for i, tree in enumerate(self.trees):
                predictions_per_class[self.make_prediction(X[j], tree)] += 1
            predictions[j] = np.argmax(predictions_per_class)

        return predictions
