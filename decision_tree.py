import numpy as np
from copy import deepcopy

class DecisionTree:
    '''Decision Tree Classifier.

    Note that this class only supports binary classification.
    '''

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
        #TODO: YOUR CODE HERE
        # begin answer
        if y.size <= 1:
            return float(0)

        labels, cnt = np.unique(y, return_counts=True)
        fracs = cnt / y.size

        entropy = 0

        for frac in fracs:
            entropy += frac * np.log(frac)

        entropy = -entropy

        # end answer
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
        #TODO: YOUR CODE HERE
        # begin answer
        N, D = X.shape
        X_ent = self.entropy(y, sample_weights)
        fea_vals, val_cnt = np.unique(X[:, index], return_counts=True)


        sub_ent = 0

        for fea_val, cnt in zip(fea_vals, val_cnt):
            sample_indices = np.argwhere(np.array(X[:, index] == fea_val) == True)
            sub_ent += cnt / N * self.entropy(y[sample_indices], sample_weights[sample_indices])

        info_gain = X_ent - sub_ent

        # end answer
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
        intrinsic_val = 0

        N, D = X.shape
        _, val_cnt = np.unique(X[:, index], return_counts=True)

        for cnt in val_cnt:
            intrinsic_val += cnt / N * np.log2(cnt/N)

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

        #TODO: YOUR CODE HERE
        # begin answer
        info_gain = self._information_gain(X, y, index, sample_weights)
        intrinsic_value = self._intrinsic_value(X, y, index, sample_weights)

        info_gain_ratio = info_gain / intrinsic_value
        
        # end answer
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
        #TODO: YOUR CODE HERE
        # begin answer
        if y.size <= 1:
            return float(0)

        labels, cnt = np.unique(y, return_counts=True)
        fracs = cnt / y.size

        gini = 0

        for frac in fracs:
            gini += np.square(frac)  

        gini = 1 - gini

        # end answer
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
        #TODO: YOUR CODE HERE
        # begin answer
        N, D = X.shape

        fea_vals, val_cnt = np.unique(X[:, index], return_counts=True)

        new_impurity = 0

        for fea_val, cnt in zip(fea_vals, val_cnt):
            sample_indices = np.argwhere(
                np.array(X[:, index] == fea_val) == True)
            new_impurity += cnt / N * \
                self.gini_impurity(y[sample_indices], sample_weights[sample_indices])

        # end answer
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
        #TODO: YOUR CODE HERE
        # Hint: Do not forget to remove the index-th feature from X.
        # begin answer
        sample_indices = np.argwhere(np.array(X[:, index] == value) == True)
        sample_indices = sample_indices.ravel()
        sub_X = X[sample_indices]
        sub_y = y[sample_indices]
        sub_sample_weights = sample_weights[sample_indices]
        
        sub_X = np.delete(sub_X, index, axis=1)
        # end answer
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
        #TODO: YOUR CODE HERE
        # Note that you need to implement the sampling feature part here for random forest!
        # Hint: You may find `np.random.choice` is useful for sampling.
        # begin answer
        best_feature_idx = 0
        D = X.shape[1]

        score = np.zeros((D,))
        for d in range(D):
            score[d] = self.criterion(X, y, d, sample_weights)
        
        if self.criterion == self._gini_purification:
            best_feature_idx = np.argmin(score)
        else:    
            best_feature_idx = np.argmax(score)

        # end answer
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
        majority_label = y[0]
        #TODO: YOUR CODE HERE
        # begin answer
        from collections import Counter
        majority_label = Counter(y).most_common(1)[0][0]
        # end answer
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
            Example:
                The first best feature name is 'title', and it has 5 different values: 0,1,2,3,4. For 'title' == 4, the next best feature name is 'pclass', we continue split the remain data. If it comes to the leaf, we use the majority_label by calling majority_vote.
                mytree = {
                    'title': {
                        0: subtree0,
                        1: subtree1,
                        2: subtree2,
                        3: subtree3,
                        4: {
                            'pclass': {
                                1: majority_vote([1, 1, 1, 1]) # which is 1, majority_label
                                2: majority_vote([1, 0, 1, 1]) # which is 1
                                3: majority_vote([0, 0, 0]) # which is 0
                            }
                        }
                    }
                }
        """

        #TODO: YOUR CODE HERE
        # Use `_choose_best_feature` to find the best feature to split the X. Then use `_split_dataset` to
        # get subtrees.
        # Hint: You may find `np.unique` is useful.
        # begin answer

        '''My implementation
        1.  find the best feature, add its name to dict as a key
        2.  create fea_dict, set its keys as feature values
        3.  split dataset
        4.  if no other attributes in the subset
                majority vote, get the labels, add them into values of fea_dict
            elif max_depth reached or labels are all the same
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
            X_sub, y_sub, sample_weights_sub = self._split_dataset(X, y, best_fea_idx, best_fea_val, sample_weights)
            if X_sub.size == 0:
                # empty, no other attributes
                fea_dict[best_fea_val] = self.majority_vote(y_sub)
            elif (depth == self.max_depth) or (y_sub == y_sub[0]).all(): # or (X_sub == X_sub[0]).all():
                # max_depth reached, or samples in subset are all the same
                fea_dict[best_fea_val] = self.majority_vote(y_sub)
            else:
                # general case
                fea_dict[best_fea_val] = self._build_tree(X_sub, y_sub, fea_names, depth+1, sample_weights)
            
        # step 5
        mytree[best_fea_name] = fea_dict

        # end answer
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
            #TODO: YOUR CODE HERE
            # begin answer
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
            # end answer

        #TODO: YOUR CODE HERE
        # begin answer
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
        # end answer

    def show(self):
        """Plot the tree using matplotlib
        """
        if self._tree is None:
            raise RuntimeError("Estimator not fitted, call `fit` first")

        import tree_plotter
        tree_plotter.createPlot(self._tree)
