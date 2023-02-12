import numpy as np
from graphviz import Digraph

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
        # computes everything at once...
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

        leaf_corr = np.zeros((self.__n_decision_nodes,), dtype=int)
        leaf_tot = np.zeros((self.__n_decision_nodes,), dtype=int)
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
        self.__accuracy_context["leaf_totals"] = leaf_tot
        if return_computed:
            return np.nanmin(leaf_acc), tot_corr.mean()
        else:
            return leaf_acc, tot_corr

    def visualize(self, path, view=False, data_handler=None):
        data_h = data_handler if data_handler is not None else self.__model_context["data_h"]
        dot = Digraph(comment="example")

        # for d in range(depth):
        dot.node("bra0", f"[{self.__decision_features[0]}]", tooltip="tmp", shape="rect")
        for node in range(1, self.__n_branch_nodes):
            dot.node(f"bra{node}", f"[{self.__decision_features[node]}]", tooltip="tmp", shape="rect")

            parent_i = (node-1) // 2
            # edge_desc = f"< {self.__thresholds[parent_i]:.2f}" if node % 2 == 1 else f"≥ {self.__thresholds[parent_i]:.2f}"
            edge_desc = f"< {self.__model_context['b'][parent_i]:.2f}" if node % 2 == 1 else f"≥ {self.__model_context['b'][parent_i]:.2f}"
            dot.edge(f"bra{parent_i}", f"bra{node}", edge_desc)

        offset = self.__n_branch_nodes - 1
        for node, c in enumerate(self.__leaf_assignments):
            desc =  f"{self.__accuracy_context['leaf_totals'][node]} ({self.__accuracy_context['leaf_acc'][node]:.2f})"
            dot.node(f"dec{node}", desc, tooltip="tmp", shape="circle", color="red" if c == 1 else "green")#, style="filled")
            # dot.node(f"dec{node}", f"{data_h.class_mapping[c]}", tooltip="tmp", shape="circle", color="red" if c == 1 else "green", style="filled")

            parent_i = (node+offset) // 2
            # edge_desc = f"< {self.__thresholds[parent_i]:.2f}" if node % 2 == 0 else f"≥ {self.__thresholds[parent_i]:.2f}"
            edge_desc = f"< {self.__model_context['b'][parent_i]:.2f}" if node % 2 == 0 else f"≥ {self.__model_context['b'][parent_i]:.2f}"
            dot.edge(f"bra{parent_i}", f"dec{node}", edge_desc)

        dot.format = "pdf"
        dot.render(path, view=view)
