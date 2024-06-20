import numpy as np

class Node:
    """
    Node class for Decision Tree.

    Parameters
    ----------
    feature_index : int, optional
        Index of the feature to split on (internal node).
    threshold : float, optional
        Threshold value for the feature (internal node).
    left : Node, optional
        Left subtree.
    right : Node, optional
        Right subtree.
    value : int, optional
        Class label for leaf nodes.

    Attributes
    ----------
    feature_index : int
        Index of the feature to split on (internal node).
    threshold : float
        Threshold value for the feature (internal node).
    left : Node
        Left subtree.
    right : Node
        Right subtree.
    value : int
        Class label for leaf nodes.
    """
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTreeClassifier:
    """
    Decision Tree Classifier implementation.

    Parameters
    ----------
    min_samples_split : int, default=2
        Minimum number of samples required to split an internal node.
    max_depth : int or None, optional
        Maximum depth of the tree. If None, nodes are expanded until all leaves are pure or until min_samples_split is reached.
    criterion : {'gini', 'entropy'}, default='gini'
        The function to measure the quality of a split.

    Attributes
    ----------
    root : Node or None
        Root of the decision tree.
    min_samples_split : int
        Minimum number of samples required to split an internal node.
    max_depth : int or None
        Maximum depth of the tree.
    criterion : str
        The function to measure the quality of a split ('gini' or 'entropy').
    """
    def __init__(self, min_samples_split=2, max_depth=None, criterion='gini'):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion

    def build_tree(self, dataset, num_samples_per_class, curr_depth=0):
        """
        Recursively builds the decision tree.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset with features and labels.
        num_samples_per_class : np.ndarray
            Array containing the number of samples for each class.
        curr_depth : int, optional
            Current depth of the tree.

        Returns
        -------
        Node
            Root node of the built subtree.
        """
        num_labels = np.sum(num_samples_per_class != 0)
        num_samples = np.sum(num_samples_per_class)

        # Check termination conditions
        if num_labels > 1 and num_samples > self.min_samples_split and (self.max_depth is None or curr_depth < self.max_depth):
            # Find the best split
            best_split = self.get_best_split(dataset, num_samples_per_class, num_samples, num_labels)

            # If information gain is positive, continue building subtrees
            if best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], best_split['num_samples_per_class_left'], curr_depth + 1)
                right_subtree = self.build_tree(best_split['dataset_right'], best_split['num_samples_per_class_right'], curr_depth + 1)
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree)

        # Create a leaf node with the majority class label
        leaf_value = self.calculate_leaf_value(num_samples_per_class)
        return Node(value=leaf_value)

    def calculate_index(self, probability):
        """
        Calculate the impurity index (Gini or Entropy) based on the probability distribution.

        Parameters
        ----------
        probability : np.ndarray
            Probability distribution of class labels.

        Returns
        -------
        float
            Impurity index (Gini or Entropy).
        """
        if self.criterion == 'gini':
            return 1 - np.sum(probability ** 2)
        elif self.criterion == 'entropy':
            index = 0
            for p in probability:
                if p > 0:
                    index -= p * np.log2(p)
            return index
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'.")

    def calculate_leaf_value(self, num_samples_per_class):
        """
        Determine the class label for a leaf node based on the majority class.

        Parameters
        ----------
        num_samples_per_class : np.ndarray
            Array containing the number of samples for each class.

        Returns
        -------
        int
            Class label for the leaf node.
        """
        return np.argmax(num_samples_per_class)

    def get_best_split(self, dataset, num_samples_per_class, num_samples, num_labels):
        """
        Find the best split for the dataset based on the criterion.

        Parameters
        ----------
        dataset : np.ndarray
            Dataset with features and labels.
        num_samples_per_class : np.ndarray
            Array containing the number of samples for each class.
        num_samples : int
            Total number of samples in the dataset.
        num_labels : int
            Number of unique class labels.

        Returns
        -------
        dict
            Dictionary containing the best split information.
        """
        best_split = {}
        max_info_gain = -float('inf')
        parent_index = self.calculate_index(num_samples_per_class / num_samples)

        # Iterate over each feature to find the best split
        for feature_index in range(dataset.shape[1] - 1):
            # Sort dataset based on current feature
            dataset = dataset[dataset[:, feature_index].argsort()]

            # Initialize variables for tracking split points
            thresholds = dataset[0, feature_index]
            label_pointer = np.zeros(num_labels, dtype=int)
            pointer_weight = 0

            # Iterate through sorted dataset to find potential split points
            for pointer in range(num_samples):
                if dataset[pointer, feature_index] != thresholds:
                    left_child_index = self.calculate_index(label_pointer / pointer_weight)
                    right_child_index = self.calculate_index((num_samples_per_class - label_pointer) / (num_samples - pointer_weight))
                    info_gain = parent_index - (pointer_weight / num_samples) * left_child_index - ((num_samples - pointer_weight) / num_samples) * right_child_index

                    # Update best split if better info gain is found
                    if info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = thresholds
                        best_split['dataset_left'] = dataset[:pointer]
                        best_split['dataset_right'] = dataset[pointer:]
                        best_split['num_samples_per_class_left'] = label_pointer.copy()
                        best_split['num_samples_per_class_right'] = num_samples_per_class - label_pointer.copy()
                        max_info_gain = info_gain

                    thresholds = dataset[pointer, feature_index]

                label_pointer[int(dataset[pointer, -1])] += 1
                pointer_weight += 1

            best_split['info_gain'] = max_info_gain

        return best_split

    def fit(self, X, y):
        """
        Fit the decision tree classifier to the training data.

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
        # Combine features and labels into a single dataset
        dataset = np.column_stack((X, y))

        # Determine number of features
        num_samples, self.n_features = X.shape

        # Set default max_depth if not specified
        if self.max_depth is None:
            self.max_depth = num_samples

        # Calculate number of samples per class
        num_samples_per_class = np.bincount(y)

        # Build the decision tree recursively
        self.root = self.build_tree(dataset, np.array(num_samples_per_class))

        return self

    def make_prediction(self, x, tree):
        """
        Predict the class label for a single sample using the trained decision tree.

        Parameters
        ----------
        x : np.ndarray
            Single data sample of shape (n_features,).
        tree : Node
            Current node in the decision tree.

        Returns
        -------
        int
            Predicted class label.
        """
        while tree.value is None:
            if x[tree.feature_index] <= tree.threshold:
                tree = tree.left
            else:
                tree = tree.right
        return tree.value

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
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)