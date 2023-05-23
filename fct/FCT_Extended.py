try:
    from xgboost import XGBClassifier
    from skopt import BayesSearchCV
    from skopt.space import Real, Categorical, Integer
except ImportError:
    print("XGBoost or skopt is not available, related functions will fail.")
import numpy as np

from .DataHandler import DataHandler

class FCT_Extended:

    def __init__(self, classification_tree, data_handler, seed=0, cv_folds=3, search_iterations=100, context=None):
        self.param_distributions = {
            "max_depth": Integer(1, 7),
            "min_child_weight": Integer(1, 1e2, prior="log-uniform"),
            "learning_rate": Real(1e-5, 0.7, prior="log-uniform"),
            "subsample": Real(0.5, 1),
            "colsample_bylevel": Real(0.5, 1),
            "colsample_bytree": Real(0.5, 1),
            "gamma": Real(1e-8, 7, prior="log-uniform"), # min_split_loss
            "reg_alpha": Real(1e-8, 1e2, prior="log-uniform"), # L1 regularization
            "reg_lambda": Real(1,4, prior="log-uniform"), # L2 regularization
            "n_estimators": Integer(10, 500), # don't want too many to be able to compute...

            # for binary
            "objective": Categorical(["binary:logistic"]),
            # for multiclass: (do not use it yet, benchmark datasets are all binary)
            # "objective": Categorical(["multi:softmax"]),
            # "num_class": Categorical([n_classes]),
        }
        if context is not None:
            self.__fct = context["classification_tree"]
            dh_ctx = context["data_h_setup"]
            data_h = DataHandler(dh_ctx["path"], dh_ctx["round_limit"])
            data_h.get_training_data(dh_ctx["split_seed"], dh_ctx["test_size"], dh_ctx["limit"], True)
            self.__data_h = data_h
            self.__params_in_leaves = context["leaf_params"]

            if "leaf_models" in context:
                self.__models_in_leaves = context["leaf_models"]
            else:
                self.__models_in_leaves = {}
                X, y = data_h.used_data # USE ALL TRAINING DATA HERE? currently not, to measure up to previous works
                X_reduced = data_h.unnormalize(data_h.normalize(X)) # performs better with true data
                for leaf_i, indices, pred in classification_tree.get_leafs_with_data(X_reduced):
                    if leaf_i in self.__params_in_leaves:
                        xgboost = XGBClassifier(random_state=dh_ctx["split_seed"], **self.__params_in_leaves[leaf_i])
                        xgboost.fit(X[indices], y[indices])
                        self.__models_in_leaves[leaf_i] = xgboost
            return

        classification_tree.reduce_tree(data_handler)
        self.__fct = classification_tree
        self.__data_h = data_handler

        X, y = data_handler.used_data # USE ALL TRAINING DATA HERE? currently not, to measure up to previous works
        X_reduced = data_handler.unnormalize(data_handler.normalize(X)) # performs better with true data
        self.__models_in_leaves = {}
        self.__params_in_leaves = {}
        for leaf_i, indices, pred in classification_tree.get_leafs_with_data(X_reduced):
            # if the leaf is not perfectly classified:
            if np.any(y[indices] != pred):
                classes, counts = np.unique(y[indices], return_counts=True)
                if len(classes) < self.__fct.n_classes:
                    print("A leaf lacks points of one class, skipped")
                    print(data_handler.get_setup(), leaf_i)
                    print(f"Has classes {classes}, is classified as {pred}")
                    continue
                if np.all(counts >= cv_folds):
                    clf = XGBClassifier(random_state=seed)
                    bayes_search = BayesSearchCV(clf, self.param_distributions, n_iter=search_iterations, # specify how many iterations
                                                    scoring="accuracy", n_jobs=8, cv=cv_folds, random_state=seed)
                    bayes_search.fit(X[indices], y[indices])
                    best_params = bayes_search.best_params_
                else:
                    # if you cannot do reasonable cross validation, optimize with some simple paramteters
                    best_params = {"max_depth": 5, "n_estimators": 1, "objective": "binary:logistic"}

                xgboost = XGBClassifier(random_state=seed, **best_params)
                xgboost.fit(X[indices], y[indices])
                self.__models_in_leaves[leaf_i] = xgboost
                self.__params_in_leaves[leaf_i] = best_params

    def compute_accuracy(self, X, y, soft_limit=0):
        # assigned = np.empty_like(y, dtype=int)
        correct = np.empty_like(y, dtype=bool)
        X_reduced = self.__data_h.unnormalize(self.__data_h.normalize(X))
        for leaf_i, indices, pred in self.__fct.get_leafs_with_data(X):
            # assigned[indices] = leaf_i
            if leaf_i in self.__models_in_leaves:
                correct[indices] = y[indices] == self.__models_in_leaves[leaf_i].predict(X[indices])
            else:
                correct[indices] = y[indices] == pred
        leaf_acc, _ = self.__fct.compute_leaf_accuracy_reduced(X, y, soft_limit=soft_limit)
        return leaf_acc, correct.mean()

    def get_context(self):
        return {
            "classification_tree": self.__fct,
            "data_h_setup": self.__data_h.get_setup(),
            "leaf_params": self.__params_in_leaves,
            "leaf_models": self.__models_in_leaves,
        }

    @staticmethod
    def create_from_context(ctx):
        return FCT_Extended(None, None, context=ctx)
