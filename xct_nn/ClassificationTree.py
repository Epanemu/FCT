import numpy as np

class ClassificationTree:
    def __init__(self, model_context, decision_features, thresholds, leaf_assignments):
        self.__model_context = model_context
        self.__decision_features = decision_features
        self.__thresholds = thresholds
        self.__leaf_assignments = leaf_assignments

        self.__n_decision_nodes = leaf_assignments.shape[0]
        self.__n_branch_nodes = decision_features.shape[0]

        self.__accuracy_context = {}

    def predict(self, x):
        i = 0
        while i < self.__n_branch_nodes:
            # TODO capture node stats for visualization (how many points at each node)
            if x[self.__decision_features[i]] < self.__thresholds[i]:
                i = i*2 + 1
            else:
                i = i*2 + 2
        return self.__leaf_assignments[i - self.__n_branch_nodes]

    def compute_accuracy(self, X, y, return_computed=True):
        acc = np.empty(y.shape, dtype=bool)
        for i, (x, true_class) in enumerate(zip(X, y)):
            acc[i] = self.predict(x) == true_class
        if return_computed:
            return acc.mean()
        else:
            return acc

    def compute_accuracy2(self, X, y, return_computed=True):
        # significantly faster
        decisions = X[:, self.__decision_features] < self.__thresholds
        indices = np.zeros_like(y, dtype=int)
        selector = np.arange(y.shape[0])
        for _ in range(self.__model_context["depth"]):
            correct = decisions[selector, indices]
            indices[correct] = indices[correct]*2 + 1
            indices[~correct] = indices[~correct]*2 + 2

        acc = self.__leaf_assignments[indices - self.__n_branch_nodes] == y
        if return_computed:
            return acc.mean()
        else:
            return acc

    def compute_leaf_accuracy(self, X, y, return_computed=True):
        # significantly faster
        decisions = X[:, self.__decision_features] < self.__thresholds
        indices = np.zeros_like(y, dtype=int)
        i_vals = np.zeros((self.__n_branch_nodes,), dtype=int)
        selector = np.arange(y.shape[0])
        for _ in range(self.__model_context["depth"]):
            correct = decisions[selector, indices]
            for i in range(indices.min(), indices.max()+1):
                i_vals[i] += (indices == i).sum()
            indices[correct] = indices[correct]*2 + 1
            indices[~correct] = indices[~correct]*2 + 2
        self.__accuracy_context["node_visits"] = i_vals

        leaf_indices = indices - self.__n_branch_nodes
        tot_corr = self.__leaf_assignments[leaf_indices] == y

        leaf_corr = np.zeros((self.__n_decision_nodes,))
        leaf_tot = np.zeros((self.__n_decision_nodes,))
        for i in range(self.__n_decision_nodes):
            leaf_corr[i] = np.sum(tot_corr[leaf_indices == i])
            leaf_tot[i] = np.sum(leaf_indices == i)
        leaf_acc = leaf_corr/leaf_tot

        # only if i have this knowledge...
        if "hard_constraint" in self.__model_context:
            if not self.__model_context["hard_constraint"]:
                bellow_thresh = leaf_tot <= self.__model_context["leaf_acc_limit"]
                leaf_acc[bellow_thresh] = (leaf_tot[bellow_thresh] - leaf_corr[bellow_thresh]) <= self.__model_context["max_invalid"]
                self.__accuracy_context["bellow_threshold"] = bellow_thresh
        # no points in leaf, the accuracy is not influenced
        # leaf_acc[np.isnan(leaf_acc)] = 1 # it is better to know which are nans
        self.__accuracy_context["total_acc"] = tot_corr.mean()
        self.__accuracy_context["leaf_acc"] = leaf_acc
        self.__accuracy_context["leaf_total"] = leaf_tot
        if return_computed:
            return np.nanmin(leaf_acc), tot_corr.mean()
        else:
            return leaf_acc, tot_corr
