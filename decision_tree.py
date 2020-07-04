import numpy as np
from copy import deepcopy

class DecisionTree:

    def __init__(self,
                 criterion,
                 max_depth,
                 min_samples_leaf,
                 sample_feature=False):
        '''Initialize the classifier.

        Args:
            criterion (str): the criterion used to select features and split nodes.
            max_depth (int): the max depth for the decision tree. This parameter is
                a trade-off between underfitting and overfitting.
            min_samples_leaf (int): the minimal samples in a leaf. This parameter is a trade-off
                between underfitting and overfitting.
            sample_feature (bool): whether to sample features for each splitting. Note that for random forest,
                we would randomly select a subset of features for learning. Here we select sqrt(p) features.
                For single decision tree, we do not sample features.
        '''
        if criterion == 'infogain_ratio':
            self.criterion = self._information_gain_ratio
        elif criterion == 'entropy':
            self.criterion = self._information_gain
        elif criterion == 'gini':
            self.criterion = self._gini_purification
        else:
            raise Exception('Criterion should be infogain_ratio or entropy or gini')
        self._tree = None
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sample_feature = sample_feature

    def fit(self, X, y, sample_weights=None):
        """Build the decision tree according to the training data.

        Args:
            X: (pd.Dataframe) training features, of shape (N, D). Each X[i] is a training sample.
            y: (pd.Series) vector of training labels, of shape (N,). y[i] is the label for X[i], and each y[i] is
            an integer in the range 0 <= y[i] <= C. Here C = 1.
            sample_weights: weights for each samples, of shape (N,).
        """
        if sample_weights is None:
            # if the sample weights is not provided, then by default all
            # the samples have unit weights.
            sample_weights = np.ones(X.shape[0]) / X.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        feature_names = X.columns.tolist()
        X = np.array(X)
        y = np.array(y)
        self._tree = self._build_tree(X, y, feature_names, depth=1, sample_weights=sample_weights)
        return self

    @staticmethod
    def entropy(y, sample_weights):
        """Calculate the entropy for label.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the entropy for y.
        """
        entropy = 0.0

        if y.size <= 1:
            return float(0)

        # print(np.sum(sample_weights))

        labels = np.unique(y)
        # C classes
        C = labels.shape[0]
        weight_sums = np.zeros((C,))

        # print(C)

        for c in range(C):
            indices = np.argwhere(
                np.array(y == labels[c]) == True)
            weight_sums[c] = np.sum(sample_weights[indices])
            # print(weight_sums[c])
        
        # print(np.sum(weight_sums))

        for weight_sum in weight_sums:
            entropy += np.multiply(weight_sum, np.log(weight_sum))

        entropy = -entropy

        return entropy

    def _information_gain(self, X, y, index, sample_weights):
        """Calculate the information gain given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain calculated.
        """
        info_gain = 0

        # print(np.sum(sample_weights))

        N, D = X.shape
        X_ent = self.entropy(y, sample_weights)
        fea_vals, val_cnt = np.unique(X[:, index], return_counts=True)

        sub_ent = 0

        for fea_val, cnt in zip(fea_vals, val_cnt):
            sample_indices = np.argwhere(np.array(X[:, index] == fea_val) == True)
            sub_sample_weights = sample_weights[sample_indices]
            sub_weights_sum = np.sum(sub_sample_weights)
            sub_sample_weights /= sub_weights_sum
            # weight_sum of S is 1, so it's omitted
            sub_ent += sub_weights_sum * self.entropy(y[sample_indices], sub_sample_weights)

        info_gain = X_ent - sub_ent

        return info_gain
    
    def _intrinsic_value(self, X, y, index, sample_weights):
        """Calculate the intrinsic value given a vector of features

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the intrinsic value calculated.
        """

        N, D = X.shape
        fea_vals, val_cnt = np.unique(X[:, index], return_counts=True)

        intrinsic_val = 0

        for fea_val, cnt in zip(fea_vals, val_cnt):
            sample_indices = np.argwhere(
                np.array(X[:, index] == fea_val) == True)
            sub_sample_weights = sample_weights[sample_indices]
            sub_weights_sum = np.sum(sub_sample_weights)
            # weight_sum of S is 1, so it's omitted
            intrinsic_val += sub_weights_sum * np.log(sub_weights_sum)
        
        intrinsic_val = -intrinsic_val

        return intrinsic_val

    def _information_gain_ratio(self, X, y, index, sample_weights):
        """Calculate the information gain ratio given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the information gain ratio calculated.
        """
        info_gain_ratio = 0

        info_gain = self._information_gain(X, y, index, sample_weights)
        intrinsic_value = self._intrinsic_value(X, y, index, sample_weights)

        if intrinsic_value != 0:
            info_gain_ratio = info_gain / intrinsic_value
        else:
            # avoid runtime warning
            info_gain_ratio = np.Infinity

        return info_gain_ratio

    @staticmethod
    def gini_impurity(y, sample_weights):
        """Calculate the gini impurity for labels.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the gini impurity for y.
        """

        if y.size <= 1:
            return float(0)

        gini = 0

        labels = np.unique(y)
        # C classes
        C = labels.shape[0]
        weight_sums = np.zeros((C,))

        for c in range(C):
            indices = np.argwhere(
                np.array(y == labels[c]) == True)
            weight_sums[c] = np.sum(sample_weights[indices])
        
        for weight_sum in weight_sums:
            gini += np.square(weight_sum)

        gini = 1 - gini

        return gini

    def _gini_purification(self, X, y, index, sample_weights):
        """Calculate the resulted gini impurity given a vector of features.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for calculating. 0 <= index < D
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (float): the resulted gini impurity after splitting by this feature.
        """
        new_impurity = 1

        N, D = X.shape

        fea_vals, val_cnt = np.unique(X[:, index], return_counts=True)

        new_impurity = 0

        for fea_val, cnt in zip(fea_vals, val_cnt):
            sample_indices = np.argwhere(
                np.array(X[:, index] == fea_val) == True)
            sub_sample_weights = sample_weights[sample_indices]
            sub_weights_sum = np.sum(sub_sample_weights)
            sub_sample_weights /= sub_weights_sum
            new_impurity += sub_weights_sum * \
                self.gini_impurity(y[sample_indices], sub_sample_weights)

        return new_impurity

    def _split_dataset(self, X, y, index, value, sample_weights):
        """Return the split of data whose index-th feature equals value.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            index: the index of the feature for splitting.
            value: the value of the index-th feature for splitting.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (np.array): the subset of X whose index-th feature equals value.
            (np.array): the subset of y whose index-th feature equals value.
            (np.array): the subset of sample weights whose index-th feature equals value.
        """
        sub_X, sub_y, sub_sample_weights = X, y, sample_weights

        sample_indices = np.argwhere(np.array(X[:, index] == value) == True)
        sample_indices = sample_indices.ravel()
        sub_X = X[sample_indices]
        sub_y = y[sample_indices]
        sub_sample_weights = sample_weights[sample_indices]
        
        sub_X = np.delete(sub_X, index, axis=1)
        sub_sample_weights /= np.sum(sub_sample_weights)

        return sub_X, sub_y, sub_sample_weights

    def _choose_best_feature(self, X, y, sample_weights):
        """Choose the best feature to split according to criterion.

        Args:
            X: training features, of shape (N, D).
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the index for the best feature
        """

        best_feature_idx = 0
        D = X.shape[1]

        if D <= 1:
            return 0

        if self.sample_feature == False:
            if self.criterion == self._gini_purification:
                score = np.ones((D,))

                for d in range(D):
                    score[d] = self.criterion(X, y, d, sample_weights)

                best_feature_idx = np.argmin(score)
            else:
                score = np.zeros((D,))

                for d in range(D):
                    score[d] = self.criterion(X, y, d, sample_weights)

                best_feature_idx = np.argmax(score)
        else:
            sample_fea_size = np.rint(np.sqrt(D)).astype(int)
            sample_fea_indices = np.random.choice(D, size=sample_fea_size, replace=False)
            sample_fea_indices = np.sort(sample_fea_indices)

            if self.criterion == self._gini_purification:
                score = np.ones((D,))

                for i in range(sample_fea_size):
                    score[sample_fea_indices[i]] = self.criterion(
                        X, y, sample_fea_indices[i], sample_weights)

                best_feature_idx = np.argmin(score)
            else:
                score = np.zeros((D,))

                for i in range(sample_fea_size):
                    score[sample_fea_indices[i]] = self.criterion(
                        X, y, sample_fea_indices[i], sample_weights)

                best_feature_idx = np.argmax(score)

        return best_feature_idx

    @staticmethod
    def majority_vote(y, sample_weights=None):
        """Return the label which appears the most in y.

        Args:
            y: vector of training labels, of shape (N,).
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (int): the majority label
        """
        if sample_weights is None:
            sample_weights = np.ones(y.shape[0]) / y.shape[0]
        else:
            sample_weights = np.array(sample_weights) / np.sum(sample_weights)

        majority_label = y[0]

        y_signed = deepcopy(y)
        y_signed = np.where(y_signed == 0, -1, 1)

        # weighted labels
        y_signed = sample_weights * y_signed

        majority_label = np.sign(np.sum(y_signed))

        # restore label range from (-1, 1) to (0, 1)
        if majority_label > 0:
            majority_label = 1
        else:
            majority_label = 0

        return majority_label

    def _build_tree(self, X, y, feature_names, depth, sample_weights):
        """Build the decision tree according to the data.

        Args:
            X: (np.array) training features, of shape (N, D).
            y: (np.array) vector of training labels, of shape (N,).
            feature_names (list): record the name of features in X in the original dataset.
            depth (int): current depth for this node.
            sample_weights: weights for each samples, of shape (N,).

        Returns:
            (dict): a dict denoting the decision tree. 
        """

        '''My implementation
        1.  find the best feature, add its name to dict as a key
        2.  create fea_dict, set its keys as feature values
        3.  split dataset
        4.  if no other attributes in the subset
                majority vote, get the labels, add them into values of fea_dict
            elif max_depth reached or min_data_leaf reached or labels are all the same
                majority vote, get the labels, add them into values of fea_dict
            else
                build subtrees, add them into values of the fea_dict
        5.  set the fea_dict as value in the dict
        
        node = {
            fea_name : {
                val : subtree/label
                val : subtree/label
                val : subtree/label
            }
        }
        '''
        fea_names = deepcopy(feature_names)

        # step 1
        mytree = dict()
        best_fea_idx = self._choose_best_feature(X, y, sample_weights)
        best_fea_name = fea_names[best_fea_idx]
        mytree[best_fea_name] = {}
        fea_names.remove(best_fea_name)

        # step 2
        fea_dict = dict()
        best_fea_col = X[:, best_fea_idx]
        best_fea_vals = np.unique(best_fea_col)

        # step 3 and 4
        for best_fea_val in best_fea_vals:
            fea_dict[best_fea_val] = {}
            # Note: the idx-th feature has been removed from X here
            X_sub, y_sub, sample_weights_sub = self._split_dataset(
                X, y, best_fea_idx, best_fea_val, sample_weights)
            # print(np.sum(sample_weights))
            if X_sub.shape[1] == 0:
                # empty, no other attributes
                fea_dict[best_fea_val] = self.majority_vote(
                    y_sub, sample_weights=sample_weights_sub)
            elif X_sub.shape[0] < self.min_samples_leaf:
                # min_data_leaf reached
                fea_dict[best_fea_val] = self.majority_vote(
                    y_sub, sample_weights=sample_weights_sub)
            elif (depth == self.max_depth) or (y_sub == y_sub[0]).all(): # or (X_sub == X_sub[0]).all():
                # max_depth reached, or samples in subset are all the same
                fea_dict[best_fea_val] = self.majority_vote(
                    y_sub, sample_weights=sample_weights_sub)
            else:
                # general case
                fea_dict[best_fea_val] = self._build_tree(
                    X_sub, y_sub, fea_names, depth+1, sample_weights_sub)
            
        # step 5
        mytree[best_fea_name] = fea_dict

        return mytree

    def predict(self, X):
        """Predict classification results for X.

        Args:
            X: (pd.Dataframe) testing sample features, of shape (N, D).

        Returns:
            (np.array): predicted testing sample labels, of shape (N,).
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        def _classify(tree, x):
            """Classify a single sample with the fitted decision tree.

            Args:
                x: ((pd.Dataframe) a single sample features, of shape (D,).

            Returns:
                (int): predicted testing sample label.
            """

            pred_label = 0

            test_fea = (list(tree.keys()))[0]
            test_fea_dict = tree.get(test_fea)
            test_fea_vals = list(test_fea_dict.keys())

            test_fea_idx = feature_names.index(test_fea)
            x_test_fea_val = x[test_fea_idx]

            if x_test_fea_val in test_fea_vals:
                pass
            else:
                # choose a branch randomly
                import random 
                x_test_fea_val = random.choice(test_fea_vals)

            subtree = test_fea_dict.get(x_test_fea_val)
            if type(subtree) is dict:
                pred_label = _classify(deepcopy(subtree), x)
            else:
                pred_label = subtree

            return pred_label


        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

    
        N, D = X.shape
        predictions = np.zeros((N,))
        feature_names = X.columns.tolist()

        X = X.values

        dt = deepcopy(self._tree)

        for i in range(N):
            predictions[i] = _classify(dt, X[i])
        
        return predictions

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
