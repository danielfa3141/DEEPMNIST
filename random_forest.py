import numpy as np
import logging
from sklearn import tree, linear_model, ensemble
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, lil_matrix
import operator
import code
from functools import reduce

from decision_tree_with_bagging import DecisionTreeWithBaggingRegressor


class WaveletsForestRegressor:
    def __init__(self, regressor='random_forest', criterion='mse', bagging=0.8, depth=9, trees=5, features='auto',
                 seed=None):
        '''
    Construct a new 'WaveletsForestRegressor' object.

    :regressor: Regressor type. Either "rf" or "decision_tree_with_bagging". Default is "rf".
    :criterion: Splitting criterion. Same options as sklearn\'s DecisionTreeRegressor. Default is "mse".
    :bagging: Bagging. Only available when using the "decision_tree_with_bagging" regressor. Default is 0.8.
    :depth: Maximum depth of each tree. Default is 9.
    :trees: Number of trees in the forest. Default is 5.
    :features: Features to consider in each split. Same options as sklearn\'s DecisionTreeRegressor.
    :seed: Seed for random operations. Default is 2000.
    '''

        self.norms = None
        self.vals = None
        ##
        self.si_tree = None
        self.si = None
        self.feature_importances_ = None
        ##
        self.volumes = None
        self.X = None
        self.y = None
        self.rf = None

        self.regressor = regressor
        self.criterion = criterion
        self.bagging = bagging
        self.depth = depth
        self.trees = trees
        self.seed = seed

    def fit(self, X_raw, y):
        '''
    Fit non-normalized data to simplex labels.

    :X_raw: Non-normalized features, given as a 2D array with each row representing a sample.
    :y: Labels, each row is given as a vertex on the simplex.
    '''

        logging.info('Fitting %s samples' % np.shape(X_raw)[0])
        X = (X_raw - np.min(X_raw, 0)) / (np.max(X_raw, 0) - np.min(X_raw, 0))
        X = np.nan_to_num(X)
        self.X = X
        self.y = y

        regressor = None
        if self.regressor == 'decision_tree_with_bagging':
            regressor = DecisionTreeWithBaggingRegressor(
                bagging=self.bagging,
                criterion=self.criterion,
                depth=self.depth,
                trees=self.trees,
                seed=self.seed,
            )
        else:
            regressor = ensemble.RandomForestRegressor(
                n_estimators=self.trees,
                criterion=self.criterion,
                max_depth=self.depth,
                max_features='auto',
                n_jobs=-1,
                random_state=self.seed,
            )

        rf = regressor.fit(X, y)
        self.rf = rf

        try:
            val_size = np.shape(y)[1]
        except:
            val_size = 1
        self.norms = np.array([])
        self.vals = np.zeros((val_size, 0))
        self.volumes = np.array([])
        ##
        self.si_tree = np.zeros((np.shape(X)[1], len(rf.estimators_)))
        self.si = np.array([])
        ##

        for i in range(len(rf.estimators_)):
            logging.info('Working on tree %s' % i)
            estimator = rf.estimators_[i]
            num_nodes = len(estimator.tree_.value)
            num_features = np.shape(X)[1]
            node_box = np.zeros((num_nodes, num_features, 2))
            node_box[:, :, 1] = 1

            norms = np.zeros(num_nodes)
            vals = np.zeros((val_size, num_nodes))
            self.__traverse_nodes(estimator, 0, node_box, norms, vals)

            logging.info('Traversing nodes of tree %s to extract volumes and norms' % i)
            volumes = np.product(node_box[:, :, 1] - node_box[:, :, 0], 1)
            norms = np.multiply(norms, np.sqrt(volumes))

            logging.info('Number of wavelets in tree %s: %s' % (i, np.shape(norms)[0]))

            self.volumes = np.append(self.volumes, volumes)
            self.norms = np.append(self.norms, norms)
            self.vals = np.append(self.vals, vals, axis=1)
            ##
            for k in range(0, num_features):
                vi_node_box = np.zeros((num_nodes, num_features, 2))
                vi_node_box[:, :, 1] = 1
                vi_norms = np.zeros(num_nodes)
                vi_vals = np.zeros((val_size, num_nodes))
                self.__variable_importance(estimator, 0, vi_node_box, vi_norms, vi_vals, k)
                self.si_tree[k, i] = np.sum(vi_norms)

        self.si = np.append(self.si, np.sum(self.si_tree, 1) / len(rf.estimators_))
        self.feature_importances_ = self.si
        ##

        return self

    def __compute_norm(self, avg, parent_avg, volume):
        norm = np.sqrt(np.sum(np.square(avg - parent_avg)) * volume)
        return norm

    def __traverse_nodes(self, estimator, base_node_id, node_box, norms, vals):
        if base_node_id == 0:
            vals[:, base_node_id] = estimator.tree_.value[base_node_id][:, 0]
            norms[base_node_id] = self.__compute_norm(vals[:, base_node_id], 0, 1)

        left_id = estimator.tree_.children_left[base_node_id]
        right_id = estimator.tree_.children_right[base_node_id]
        if left_id >= 0:
            node_box[left_id, :, :] = node_box[base_node_id, :, :]
            node_box[left_id, estimator.tree_.feature[base_node_id], 1] = np.min(
                [estimator.tree_.threshold[base_node_id], node_box[left_id, estimator.tree_.feature[base_node_id], 1]])
            self.__traverse_nodes(estimator, left_id, node_box, norms, vals)
            vals[:, left_id] = estimator.tree_.value[left_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
            norms[left_id] = self.__compute_norm(vals[:, left_id], vals[:, base_node_id], 1)
        if right_id >= 0:
            node_box[right_id, :, :] = node_box[base_node_id, :, :]
            node_box[right_id, estimator.tree_.feature[base_node_id], 0] = np.max(
                [estimator.tree_.threshold[base_node_id], node_box[right_id, estimator.tree_.feature[base_node_id], 0]])
            self.__traverse_nodes(estimator, right_id, node_box, norms, vals)
            vals[:, right_id] = estimator.tree_.value[right_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
            norms[right_id] = self.__compute_norm(vals[:, right_id], vals[:, base_node_id], 1)

    ##
    def __variable_importance(self, estimator, base_node_id, vi_node_box, vi_norms, vi_vals, feature):
        if base_node_id == 0:
            vi_vals[:, base_node_id] = estimator.tree_.value[base_node_id][:, 0]
            vi_norms[base_node_id] = self.__compute_norm(vi_vals[:, base_node_id], 0, 1)

        left_id = estimator.tree_.children_left[base_node_id]
        right_id = estimator.tree_.children_right[base_node_id]
        if left_id >= 0:
            vi_node_box[left_id, :, :] = vi_node_box[base_node_id, :, :]
            vi_node_box[left_id, estimator.tree_.feature[base_node_id], 1] = np.min(
                [estimator.tree_.threshold[base_node_id],
                 vi_node_box[left_id, estimator.tree_.feature[base_node_id], 1]])
            self.__variable_importance(estimator, left_id, vi_node_box, vi_norms, vi_vals, feature)
            vi_vals[:, left_id] = estimator.tree_.value[left_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
            if estimator.tree_.feature[estimator.tree_.children_left[base_node_id]] == feature:
                vi_norms[left_id] = self.__compute_norm(vi_vals[:, left_id], vi_vals[:, base_node_id], 1)
        if right_id >= 0:
            vi_node_box[right_id, :, :] = vi_node_box[base_node_id, :, :]
            vi_node_box[right_id, estimator.tree_.feature[base_node_id], 0] = np.max(
                [estimator.tree_.threshold[base_node_id],
                 vi_node_box[right_id, estimator.tree_.feature[base_node_id], 0]])
            self.__variable_importance(estimator, right_id, vi_node_box, vi_norms, vi_vals, feature)
            vi_vals[:, right_id] = estimator.tree_.value[right_id][:, 0] - estimator.tree_.value[base_node_id][:, 0]
            if estimator.tree_.feature[estimator.tree_.children_right[base_node_id]] == feature:
                vi_norms[right_id] = self.__compute_norm(vi_vals[:, right_id], vi_vals[:, base_node_id], 1)

    ##

    def predict(self, X, m=1000, start_m=0, paths=None):
        '''
    Predict using a maximum of M-terms

    :X: Data samples.
    :m: Maximum of M-terms.
    :start_m: The index of the starting term. Can be used to evaluate predictions incrementally over terms.
    :paths: Instead of computing decision paths for each sample, the method can receive the indicator matrix. Can be used to evaluate predictions incrementally over terms.
    :return: Predictions.
    '''

        sorted_norms = np.argsort(-self.norms)[start_m:m]
        if paths == None:
            paths, n_nodes_ptr = self.rf.decision_path(X)
        pruned = lil_matrix(paths.shape, dtype=np.float32)
        pruned[:, sorted_norms] = paths[:, sorted_norms]
        predictions = pruned * self.vals.T / len(self.rf.estimators_)

        return predictions

    def evaluate_smoothness(self, m=1000):
        '''
    Evaluates smoothness for a maximum of M-terms
    
    :m: Maximum terms to use. Default is 1000.
    :return: Smothness index, n_wavelets, errors.
    '''
        n_wavelets = []
        errors = []
        step = 10
        power = 2

        paths, n_nodes_ptr = self.rf.decision_path(self.X)
        predictions = np.zeros(np.shape(self.y))
        for m_step in range(2, m, step):
            if m_step > len(self.norms):
                break
            predictions += self.predict(self.X, m_step, max(m_step - step, 0), paths=paths)
            error_norms = np.power(np.sum(np.power(self.y - predictions, power), 1), 1. / power)
            total_error = np.sum(np.square(error_norms), 0) / len(self.X)
            if m_step > 0 and total_error > 0:
                logging.info('Error for m=%s: %s' % (m_step - 1, total_error))
                n_wavelets.append(m_step - 1)
                errors.append(total_error)

        n_wavelets_log = np.log(np.reshape(n_wavelets, (-1, 1)))
        errors_log = np.log(np.reshape(errors, (-1, 1)))

        regr = linear_model.LinearRegression()
        regr.fit(n_wavelets_log, errors_log)
        alpha = np.abs(regr.coef_[0][0])
        logging.info('Smoothness index: %s' % alpha)

        return alpha, n_wavelets, errors

    def accuracy(self, y_pred, y):
        '''
    Evaluates accuracy given predictions and actual labels. 
    
    :y_pred: Predictions as vertices on the simplex (preprocessed by 'pred_to_one_hot').
    :y: Actual labels.
    :return: Accuracy.
    '''
        return accuracy_score(y, y_pred)

    def pred_to_one_hot(self, y_pred):
        '''
    Converts regression predictions to their closest vertices on the simplex
    '''
        argmax = np.argmax(y_pred, 1)
        ret = np.zeros((len(argmax), np.shape(y_pred)[1]))
        ret[np.arange(len(argmax)), argmax] = 1
        return ret
