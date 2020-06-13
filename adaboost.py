import copy
import numpy as np
from decision_tree import DecisionTree


class Adaboost:
    '''Adaboost Classifier.

    Note that this class only support binary classification.
    '''

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
        self._alphas = [1 for _ in range(n_estimator)]

    def fit(self, X, y):
        """Build the Adaboost according to the training data.

        Args:
            X: training features, of shape (N, D). Each X[i] is a training sample.
            y: vector of training labels, of shape (N,).
        """
        #TODO: YOUR CODE HERE

        # begin answer
        N, D = X.shape

        # initial weights -> 1/N
        weights = np.full((N,), np.divide(1, N))

        idx = 0

        for estimator in self._estimators:
            # grow a weak learner
            estimator.fit(X, y, sample_weights=weights)
            # make predictions on training set
            y_pred = estimator.predict(X)
            # get error rate
            weighted_loss = np.sum(weights * np.where(y == y_pred, 0, 1))
            error_rate = np.divide(weighted_loss, np.sum(weights))
            # update alpha i.e. log odd
            self._alphas[idx] = np.log(np.divide(1-error_rate, error_rate))
            # update weights
            weights = weights * \
                np.exp(self._alphas[idx] * np.where(y == y_pred, 0, 1))
            
            idx += 1

        # end answer
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
        #TODO: YOUR CODE HERE
        # begin answer
        N, _ = X.shape
        predictions = np.zeros((self.n_estimator, N))

        idx = 0

        for estimator in self._estimators:
            predictions[idx] = estimator.predict(X)
            idx += 1

        for i in range(N):
            y_pred[i] = DecisionTree.majority_vote(
                predictions[:, i].reshape(self.n_estimator,), 
                sample_weights=np.array(self._alphas))

        # end answer
        return y_pred
