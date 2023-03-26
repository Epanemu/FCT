from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

from xct_nn.DataHandler import DataHandler

class XCT_Extended:
    # param_distributions = {
    #     # "booster": ["gbtree", "gblinear", "dart"], # gblinear is not a tree model
    #     "booster": ["gbtree", "dart"],
    #     "max_depth": [3, 4, 5, 6, 7, 8, 9],
    #     "min_child_weight": [0.1, 0.5, 1, 2, 5],
    #     "max_leaves": [0, 5, 10, 15],
    #     "max_delta_step": [0, 1, 5], # not needed
    #     "min_split_loss": [0, 0.01, 0.05, 0.1],
    #     # "reg_alpha": [0] # L1 regularization
    #     # "reg_lambda": [1] # L2 regularization
    #     "n_estimators": [20, 50, 100, 200, 500], # 1000 is too many to be able to compute...

    #     # for binary
    #     "objective": ["binary:logistic"],
    #     # for multiclass:
    #     # "objective": ["multi:softmax"],
    #     # "num_class": [n_classes],
    # }

    param_distributions = {
        "booster": ["gbtree"],
        "max_depth": [8],
        "n_estimators": [200],
        "objective": ["binary:logistic"],
    }

    def __init__(self, classification_tree, data_handler, seed=0, cv_folds=3, search_iterations=100, context=None):
        if context is not None:
            self.__xct = context["classification_tree"]
            dh_ctx = context["data_h_setup"]
            data_h = DataHandler(dh_ctx["path"], dh_ctx["round_limit"])
            data_h.get_training_data(dh_ctx["split_seed"], dh_ctx["test_size"], dh_ctx["limit"], True)
            self.__data_h = data_h
            self.__params_in_leaves = context["leaf_params"]

            self.__models_in_leaves = {}
            X, y = data_h.train_data # USE ALL TRAINING DATA HERE
            X_reduced = data_h.unnormalize(data_h.normalize(X)) # performs better with true data
            for leaf_i, indices, pred in classification_tree.get_leafs_with_data(X_reduced):
                if leaf_i in self.__params_in_leaves:
                    xgboost = XGBClassifier(random_state=dh_ctx["split_seed"], **self.__params_in_leaves[leaf_i])
                    xgboost.fit(X[indices], y[indices])
                    self.__models_in_leaves[leaf_i] = xgboost
            return

        classification_tree.reduce_tree(data_handler)
        self.__xct = classification_tree
        self.__data_h = data_handler

        X, y = data_handler.train_data # USE ALL TRAINING DATA HERE
        X_reduced = data_handler.unnormalize(data_handler.normalize(X)) # performs better with true data
        self.__models_in_leaves = {}
        self.__params_in_leaves = {}
        for leaf_i, indices, pred in classification_tree.get_leafs_with_data(X_reduced):
            # if the leaf is not perfectly classified:
            if np.any(y[indices] != pred):
                classes, counts = np.unique(y[indices], return_counts=True)
                if len(classes) < self.__xct.n_classes:
                    print("A leaf lacks points of one class, skipped")
                    print(data_handler.get_setup(), leaf_i)
                    print(f"Has classes {classes}, is classified as {pred}")
                    continue
                if np.all(counts >= cv_folds):
                    clf = XGBClassifier(random_state=seed)
                    search = RandomizedSearchCV(clf, self.param_distributions, cv=cv_folds, n_iter=search_iterations, random_state=seed)
                    search = search.fit(X[indices], y[indices])
                    best_params = search.best_params_
                else:
                    # if you cannot do cross validation, optimize with some simple paramteters
                    best_params = {"booster": "dart", "max_depth": 5, "n_estimators": 5, "objective": "binary:logistic"}

                xgboost = XGBClassifier(random_state=seed, **best_params)
                xgboost.fit(X[indices], y[indices])
                self.__models_in_leaves[leaf_i] = xgboost
                self.__params_in_leaves[leaf_i] = best_params

    def compute_accuracy(self, X, y):
        # assigned = np.empty_like(y, dtype=int)
        correct = np.empty_like(y, dtype=bool)
        X_reduced = self.__data_h.unnormalize(self.__data_h.normalize(X))
        for leaf_i, indices, pred in self.__xct.get_leafs_with_data(X):
            # assigned[indices] = leaf_i
            if leaf_i in self.__models_in_leaves:
                correct[indices] = y[indices] == self.__models_in_leaves[leaf_i].predict(X[indices])
            else:
                correct[indices] = y[indices] == pred
        leaf_acc, _ = self.__xct.compute_leaf_accuracy_reduced(X, y)
        return leaf_acc, correct.mean()

    def get_context(self):
        return {
            "classification_tree": self.__xct,
            "data_h_setup": self.__data_h.get_setup(),
            "leaf_params": self.__params_in_leaves[leaf_i],
        }

    @staticmethod
    def create_from_context(ctx):
        return XCT_Extended(None, None, context=ctx)
