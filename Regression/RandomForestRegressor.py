import numpy as np
from DecisionTreeRegressor import DecisionTreeRegressor, Node

class RandomForestRegressor(DecisionTreeRegressor):
    def __init__(self, n_estimators: int = 100, min_samples_split: int = 2, max_depth: int = None, 
                 tol: float = 1e-6, max_features: str = 'auto', bootstrap_fraction: float = None):
        """
        Random Forest Regressor constructor.

        Parameters
        ----------
        n_estimators : int, default=100
            The number of trees in the forest.
        
        min_samples_split : int, default=2
            The minimum number of samples required to split an internal node.
        
        max_depth : int or None, default=None
            The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain
            less than min_samples_split samples.
        
        tol : float, default=1e-6
            Tolerance for variance-based stopping criterion.
        
        max_features : {'auto', 'sqrt', 'log2'}, int or float, default='auto'
            The number of features to consider when looking for the best split:
            - If 'auto', then max_features = n_features.
            - If 'sqrt', then max_features = sqrt(n_features).
            - If 'log2', then max_features = log2(n_features).
        
        bootstrap_fraction : float or None, default=None
            The fraction of samples to use for bootstrap resampling. If None, it uses 1.0 (all samples).
        """
        super().__init__(min_samples_split, max_depth, tol)
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.bootstrap_fraction = bootstrap_fraction
        self.trees = np.empty(self.n_estimators, dtype=object)

    def bootstrap_resample(self, dataset: np.ndarray) -> np.ndarray:
        """
        Create a bootstrap sample from the dataset.

        Parameters
        ----------
        dataset : np.ndarray
            The original dataset to sample from.

        Returns
        -------
        np.ndarray
            The bootstrapped dataset.
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Random Forest Regressor on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.
        
        y : np.ndarray of shape (n_samples,)
            The target values.
        """
        dataset = np.column_stack((X, y))
        self.num_samples, self.features = X.shape

        if self.max_features == 'sqrt':
            self.max_features = int(np.sqrt(self.features))
        elif self.max_features == 'log2':
            self.max_features = int(np.log2(self.features))
        elif self.max_features == 'auto':
            self.max_features = self.features

        self.n_features = self.max_features

        if self.max_depth is None:
            self.max_depth = self.num_samples

        for i in range(self.n_estimators):
            bootstrapped_sample = self.bootstrap_resample(dataset)
            target = bootstrapped_sample[:, -1]
            score_matrix = np.array([self.num_samples, np.sum(target), np.sum(target ** 2)])
            self.trees[i] = self.build_tree(bootstrapped_sample, score_matrix)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for the given input data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        np.ndarray
            The predicted target values.
        """
        num_samples = X.shape[0]
        predictions = np.zeros(num_samples)

        for i in range(num_samples):
            for tree in self.trees:
                predictions[i] += self.make_prediction(X[i], tree)
            predictions[i] /= self.n_estimators

        return predictions