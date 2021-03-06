import copy
import numpy as np
from decision_tree import DecisionTree


class RandomForest:

    def __init__(self,
                 base_learner,
                 n_estimator,
                 seed=2020):
        '''Initialize the classifier.

        Args:
            base_learner: the base_learner should provide the .fit() and .predict() interface.
            n_estimator (int): The number of base learners in RandomForest.
            seed (int): random seed
        '''
        np.random.seed(seed)
        self.base_learner = base_learner
        self.n_estimator = n_estimator
        self._estimators = [copy.deepcopy(self.base_learner) for _ in range(self.n_estimator)]

    def _get_bootstrap_dataset(self, X, y):
        """Create a bootstrap dataset for X.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).

        Returns:
            X_bootstrap: a sampled dataset, of shape (N, D).
            y_bootstrap: the labels for sampled dataset.
        """

        N, D = X.shape
        X_bootstrap = np.zeros((N, D))
        y_bootstrap = np.zeros((N,))

        choice_indices = np.random.choice(N, size=N, replace=True)
        choice_indices = np.sort(choice_indices)

        X_bootstrap = X.iloc[choice_indices]
        y_bootstrap = y.iloc[choice_indices]

        return X_bootstrap, y_bootstrap

    def fit(self, X, y):
        """Build the random forest according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """

        # grow the trees
        for estimator in self._estimators:
            X_b, y_b = self._get_bootstrap_dataset(X, y)
            estimator.fit(X_b, y_b)

        return self

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        N = X.shape[0]
        y_pred = np.zeros(N)

        N, _ = X.shape
        predictions = np.zeros((self.n_estimator, N))

        idx = 0

        for estimator in self._estimators:
            predictions[idx] = estimator.predict(X)
            idx += 1
        
        for i in range(N):
            y_pred[i] = DecisionTree.majority_vote(predictions[:, i].reshape(self.n_estimator,))

        return y_pred
