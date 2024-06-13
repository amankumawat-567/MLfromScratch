import numpy as np

class Node:
    def __init__(self, feature_index: int = None, threshold: float = None, left: 'Node' = None, right: 'Node' = None, value: float = None):
        """
        Node class represents a node in a decision tree.

        Parameters
        ----------
        feature_index : int, optional
            The index of the feature used for splitting at this node.

        threshold : float, optional
            The threshold value for splitting at this node.

        left : Node, optional
            The left child node.

        right : Node, optional
            The right child node.

        value : float, optional
            The predicted value at a leaf node.
        """
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=None, tol=1e-6):
        """
        Decision Tree Regressor constructor.

        Parameters
        ----------
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        
        max_depth : int or None, default=None
            The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure
            or until all leaves contain less than min_samples_split samples.
        
        tol : float, default=1e-6
            Tolerance for variance-based stopping criterion.
        """
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.tol = tol

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.

        y : np.ndarray of shape (n_samples,)
            The target values.

        Raises
        ------
        ValueError
            If the length of X and y do not match.
        """
        if len(X) != len(y):
            raise ValueError("Length of X and y must be the same.")
        
        dataset = np.column_stack((X, y))
        num_samples, self.n_features = X.shape
        
        if self.max_depth is None:
            self.max_depth = num_samples
        
        Score_Matrix = np.array([num_samples, np.sum(y), np.sum(y ** 2)])
        self.root = self.build_tree(dataset, Score_Matrix)

    def build_tree(self, dataset: np.ndarray, Score_Matrix: np.ndarray, curr_depth: int = 0) -> Node:
        """
        Recursively builds the decision tree.

        Parameters
        ----------
        dataset : np.ndarray
            The dataset to split.

        Score_Matrix : np.ndarray
            The score matrix computed from the dataset.

        curr_depth : int, default=0
            The current depth of the tree.

        Returns
        -------
        Node
            The root node of the constructed tree.
        """
        num_samples = Score_Matrix[0]
        variance = self.compute_variance(Score_Matrix)
        
        if variance > self.tol and num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, Score_Matrix)
            
            if best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], best_split['Score_Matrix_left'], curr_depth + 1)
                right_subtree = self.build_tree(best_split['dataset_right'], best_split['Score_Matrix_right'], curr_depth + 1)
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree)
        
        leaf_value = self.calculate_leaf_value(Score_Matrix)
        return Node(value=leaf_value)

    def get_best_split(self, dataset: np.ndarray, Score_Matrix: np.ndarray) -> dict:
        """
        Finds the best split for the current dataset.

        Parameters
        ----------
        dataset : np.ndarray
            The dataset to split.

        Score_Matrix : np.ndarray
            The score matrix computed from the dataset.

        Returns
        -------
        dict
            A dictionary containing the best split information.
        """
        best_split = {}
        max_info_gain = -float("inf")
        parent_variance = self.compute_variance(Score_Matrix)
        
        for feature_index in range(self.n_features):
            dataset_sorted = dataset[dataset[:, feature_index].argsort()]
            thresholds = dataset_sorted[0, feature_index]
            score_pointer = np.zeros(3)
            
            for pointer in range(int(Score_Matrix[0])):
                if dataset_sorted[pointer, feature_index] != thresholds:
                    left_child_variance = self.compute_variance(score_pointer)
                    right_child_variance = self.compute_variance(Score_Matrix - score_pointer)
                    
                    info_gain = parent_variance - (score_pointer[0] / Score_Matrix[0]) * left_child_variance + ((Score_Matrix[0] - score_pointer[0]) / Score_Matrix[0]) * right_child_variance
                    
                    if info_gain >= max_info_gain:
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': thresholds,
                            'dataset_left': dataset_sorted[:pointer],
                            'Score_Matrix_left': score_pointer.copy(),
                            'dataset_right': dataset_sorted[pointer:],
                            'Score_Matrix_right': Score_Matrix - score_pointer.copy()
                        }
                        max_info_gain = info_gain
                    
                    thresholds = dataset_sorted[pointer, feature_index]
                
                score_pointer[0] += 1
                score_pointer[1] += dataset_sorted[pointer, -1]
                score_pointer[2] += dataset_sorted[pointer, -1] ** 2

            best_split['info_gain'] = max_info_gain
        
        return best_split

    def compute_variance(self, Score_Matrix: np.ndarray) -> float:
        """
        Computes the variance of the given score matrix.

        Parameters
        ----------
        Score_Matrix : np.ndarray
            The score matrix.

        Returns
        -------
        float
            The computed variance.
        """
        if Score_Matrix[0] == 0:
            return 0
        
        mean = Score_Matrix[1] / Score_Matrix[0]
        variance = Score_Matrix[2] / Score_Matrix[0] - mean ** 2
        return variance

    def calculate_leaf_value(self, Score_Matrix: np.ndarray) -> float:
        """
        Calculates the leaf value for a node.

        Parameters
        ----------
        Score_Matrix : np.ndarray
            The score matrix.

        Returns
        -------
        float
            The calculated leaf value.
        """
        if Score_Matrix[0] != 0:
            return Score_Matrix[1] / Score_Matrix[0]
        
        return 0

    def make_prediction(self, x: np.ndarray, tree: Node) -> float:
        """
        Makes a prediction using the decision tree.

        Parameters
        ----------
        x : np.ndarray
            The input data point to predict.

        tree : Node
            The root node of the decision tree.

        Returns
        -------
        float
            The predicted value.
        """
        while tree.value is None:
            if x[tree.feature_index] <= tree.threshold:
                tree = tree.left
            else:
                tree = tree.right
        
        return tree.value

    def predict(self, X: np.ndarray) -> list:
        """
        Predicts target values for the input data.

        Parameters
        ----------
        X : np.ndarray
            The input data.

        Returns
        -------
        list
            The predicted target values.
        """
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)
