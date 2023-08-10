from sklearn import tree as skltree
from sklearn.model_selection import GridSearchCV
import pickle
import sys

from xct_nn.DataHandler import DataHandler
from xct_nn.TreeGenerator import TreeGenerator
from xct_nn.XCT_Extended import XCT_Extended
from utils.datasets import DATASETS_INFO

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


PARAMETERS = {
    "base": {
        "search_space": {
            "max_depth": Integer(2, 4),
            "min_samples_leaf": Integer(1, 60),
            "min_samples_split": Integer(2, 60),
            "max_leaf_nodes": Integer(8, 16),
            "min_impurity_decrease": Real(0, 0.2),
        },
        "round_lim": 40,
        "size_lim": 10000,
    },
    "d3": {
        "search_space": {
            "max_depth": Integer(1, 3),
            "min_samples_leaf": Integer(1, 60),
            "min_samples_split": Integer(2, 60),
            "max_leaf_nodes": Integer(2, 8),
            "min_impurity_decrease": Real(0, 0.2),
        },
        "round_lim": 40,
        "size_lim": 10000,
    },
    "unconstrained": {
        "search_space": {
            "max_depth": Integer(2, 20),
            "min_samples_leaf": Integer(1, 60),
            "min_samples_split": Integer(2, 60),
            "max_leaf_nodes": Integer(8, 512),
            "min_impurity_decrease": Real(0, 0.2),
        },
        "round_lim": 40,
        "size_lim": 10000,
    },
}

for name, params in PARAMETERS.items():
    data_values = {}
    for dtype in DATASETS_INFO:
        for dataname in DATASETS_INFO[dtype]:
            dataset_path = DATASETS_INFO[dtype][dataname]["path"]
            data_id = f"{dtype[0]}_{dataname}"
            data_values[data_id] = {
                "train_accs": [],
                "train_leaf_accs": [],
                "train_leaf_soft_accs": [],
                "ext_train_accs": [],
                "ext_train_leaf_accs": [],
                "test_accs": [],
                "test_leaf_accs": [],
                "test_leaf_soft_accs": [],
                "ext_test_accs": [],
                "ext_test_leaf_accs": [],
                "best_params": [],
            }
            for seed in range(10):
                data_handler = DataHandler(dataset_path, round_limit=params["round_lim"])
                X_train, y_train = data_handler.get_training_data(split_seed=seed, test_size=0.2, limit=params["size_lim"])

                search_space = params["search_space"]

                skl_tree = skltree.DecisionTreeClassifier(random_state=seed)
                bayes_search = BayesSearchCV(skl_tree, search_space, n_iter=100, # specify how many iterations
                                                    scoring="accuracy", n_jobs=8, cv=5, random_state=seed)

                bayes_search.fit(X_train, y_train) # callback=on_step will print score after each iteration
                data_values[data_id]["best_params"].append(bayes_search.best_params_)
                dt_sklearn = skltree.DecisionTreeClassifier(random_state=seed, **bayes_search.best_params_)

                dt_sklearn.fit(X_train, y_train)

                tree_gen = TreeGenerator(data_handler)
                tree = tree_gen.make_from_sklearn(dt_sklearn.tree_, soft_limit=0, train_data=X_train)
                ext_tree = XCT_Extended(tree, data_handler, seed, search_iterations=50)
                X, y = data_handler.used_data
                # necessary for rounding
                X = data_handler.unnormalize(data_handler.normalize(X))
                lacc_soft, acc = tree.compute_leaf_accuracy(X, y, soft_limit=20)
                lacc_hard, _ = tree.compute_leaf_accuracy(X, y)
                data_values[data_id]["train_leaf_accs"].append(lacc_hard)
                data_values[data_id]["train_leaf_soft_accs"].append(lacc_soft)
                data_values[data_id]["train_accs"].append(acc)
                ext_leaf, ext = ext_tree.compute_accuracy(X, y)
                data_values[data_id]["ext_train_leaf_accs"].append(ext_leaf) # leaf accuracy of reduced tree is stored in the extended leaf accuracy
                data_values[data_id]["ext_train_accs"].append(ext)

                X, y = data_handler.test_data
                # necessary for rounding
                X = data_handler.unnormalize(data_handler.normalize(X))
                lacc_soft, acc = tree.compute_leaf_accuracy(X, y, soft_limit=20)
                lacc_hard, _ = tree.compute_leaf_accuracy(X, y)
                data_values[data_id]["test_leaf_accs"].append(lacc_hard)
                data_values[data_id]["test_leaf_soft_accs"].append(lacc_soft)
                data_values[data_id]["test_accs"].append(acc)
                ext_leaf, ext = ext_tree.compute_accuracy(X, y)
                data_values[data_id]["ext_test_leaf_accs"].append(ext_leaf) # leaf accuracy of reduced tree is stored in the extended leaf accuracy
                data_values[data_id]["ext_test_accs"].append(ext)
    print(name)
    # with open(f"results/best_trees_ext_{name}_2bayes_seed{seed_range}_xgb.pickle", "wb") as f:
    with open(f"results/best_trees_{name}.pickle", "wb") as f:
        pickle.dump(data_values, f)
