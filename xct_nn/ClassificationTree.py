import numpy as np
try:
    from graphviz import Digraph
except ImportError:
    print("Graphviz not available, will fail if attempted to visualize a tree")

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
        leaf_tot_safe = leaf_tot.copy()
        leaf_tot_safe[leaf_tot == 0] = 1
        leaf_acc = leaf_corr/leaf_tot_safe
        leaf_acc[leaf_tot == 0] = 1 # if no points, the accuracy is 100%

        # only if i have this knowledge...
        if "hard_constraint" in self.__model_context:
            if not self.__model_context["hard_constraint"]:
                bellow_thresh = leaf_tot <= self.__model_context["leaf_acc_limit"]
                misclas = leaf_tot[bellow_thresh] - leaf_corr[bellow_thresh]
                if self.__model_context["max_invalid"] is not None:
                    leaf_acc[bellow_thresh] = misclas <= self.__model_context["max_invalid"]
                else:
                    leaf_acc[bellow_thresh] = 1 - (misclas / self.__model_context["leaf_acc_limit"])
                self.__accuracy_context["bellow_threshold"] = bellow_thresh
        # no points in leaf, the accuracy is not influenced
        # leaf_acc[np.isnan(leaf_acc)] = 1 # it is better to know which are nans
        self.__accuracy_context["total_corr"] = tot_corr
        self.__accuracy_context["leaf_acc"] = leaf_acc
        self.__accuracy_context["leaf_totals"] = leaf_tot
        if return_computed:
            return np.nanmin(leaf_acc), tot_corr.mean()
        else:
            return leaf_acc, tot_corr

    def visualize(self, path, view=False, data_handler=None, show_normalized_thresholds=True):
        data_h = data_handler if data_handler is not None else self.__model_context["data_h"]
        dot = Digraph(comment="example")

        # for d in range(depth):
        dot.node("bra0", f"[{self.__decision_features[0]}]", tooltip="tmp", shape="rect")
        for node in range(1, self.__n_branch_nodes):
            dot.node(f"bra{node}", f"[{self.__decision_features[node]}]", tooltip="tmp", shape="rect")

            parent_i = (node-1) // 2
            if show_normalized_thresholds:
                thresh = self.__model_context['b'][parent_i]
            else:
                thresh = self.__thresholds[parent_i]
            edge_desc = f"< {thresh:.2f}" if node % 2 == 1 else f"≥ {thresh:.2f}"
            dot.edge(f"bra{parent_i}", f"bra{node}", edge_desc)

        offset = self.__n_branch_nodes - 1
        for node, c in enumerate(self.__leaf_assignments):
            if "bellow_threshold" in self.__accuracy_context and self.__accuracy_context["bellow_threshold"][node]:
                desc =  f"{self.__accuracy_context['leaf_totals'][node]} [{self.__accuracy_context['leaf_acc'][node]:.2f}]"
            else:
                desc =  f"{self.__accuracy_context['leaf_totals'][node]} ({self.__accuracy_context['leaf_acc'][node]:.2f})"
            dot.node(f"dec{node}", desc, tooltip="tmp", shape="circle", color="red" if c == 1 else "green")#, style="filled")
            # dot.node(f"dec{node}", f"{data_h.class_mapping[c]}", tooltip="tmp", shape="circle", color="red" if c == 1 else "green", style="filled")

            parent_i = (node+offset) // 2
            if show_normalized_thresholds:
                thresh = self.__model_context['b'][parent_i]
            else:
                thresh = self.__thresholds[parent_i]
            edge_desc = f"< {thresh:.2f}" if node % 2 == 1 else f"≥ {thresh:.2f}"
            dot.edge(f"bra{parent_i}", f"dec{node}", edge_desc)

        dot.format = "pdf"
        dot.render(path, view=view)

    def as_warmstart(self):
        if self.__model_context is not None \
           and "a" in self.__model_context \
           and "b" in self.__model_context:
            return self.__model_context["a"], self.__model_context["b"]
        else:
            raise NotImplementedError("Case when a or b is not already present has not been considered yet.")

    @property
    def leaf_totals(self):
        if "leaf_totals" in self.__accuracy_context:
            return self.__accuracy_context["leaf_totals"]

    @property
    def leaf_accuracy(self):
        if "leaf_acc" in self.__accuracy_context:
            return self.__accuracy_context["leaf_acc"]

    @property
    def total_accuracy(self):
        if "total_corr" in self.__accuracy_context:
            return self.__accuracy_context["total_corr"].mean()

    @property
    def correct_classifications(self):
        if "total_corr" in self.__accuracy_context:
            return self.__accuracy_context["total_corr"]

    @property
    def using_soft_constraint(self):
        if "bellow_threshold" in self.__accuracy_context:
            self.__accuracy_context["bellow_threshold"]

    @property
    def depth(self):
        if "depth" in self.__model_context:
            return self.__model_context["depth"]
        else:
            return np.log2(self.__n_decision_nodes).astype(int)

    @property
    def decision_features(self):
        return self.__decision_features

    @property
    def thresholds(self):
        return self.__thresholds

    @property
    def leaf_assignments(self):
        return self.__leaf_assignments
