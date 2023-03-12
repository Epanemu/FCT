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

    def reduce_tree(self):
        mapping = [0]
        parents = [-1]
        l_child = []
        r_child = []

        q = [0]
        while q:
            i = q.pop(0)
            is_leaf = mapping[i] >= self.__model_context["b"].shape[0]
            if not is_leaf:
                orig_t = self.__model_context["b"][mapping[i]]
                while orig_t == 0:
                    mapping[i] = mapping[i]*2 + 2
                    is_leaf = mapping[i] >= self.__model_context["b"].shape[0]
                    if is_leaf:
                        break
                    orig_t = self.__model_context["b"][mapping[i]]
            if is_leaf:
                l_child.append(-1)
                r_child.append(-1)
            else:
                l_ch = len(parents)
                r_ch = l_ch + 1
                parents += [i, i]
                mapping += [mapping[i]*2 + 1, mapping[i]*2 + 2]
                q += [l_ch, r_ch]
                l_child.append(l_ch)
                r_child.append(r_ch)

        mapping = [[m] for m in mapping]

        # check for 2 same class children leafs
        pruned = []
        change = True
        while change:
            change = False
            q = [0]
            while q:
                i = q.pop(0)
                if l_child[i] != -1 and r_child[i] != -1 and l_child[l_child[i]] == -1 and l_child[r_child[i]] == -1:
                    lmap = mapping[l_child[i]][0] - self.__n_decision_nodes + 1
                    rmap = mapping[r_child[i]][0] - self.__n_decision_nodes + 1
                    if self.__leaf_assignments[rmap] == self.__leaf_assignments[lmap]:
                        mapping[i] = mapping[l_child[i]] + mapping[r_child[i]]
                        pruned += [l_child[i], r_child[i]]
                        l_child[i] = -1
                        r_child[i] = -1
                        change = True
                        break

                if l_child[i] != -1:
                    q.append(l_child[i])
                if r_child[i] != -1:
                    q.append(r_child[i])
        self.__reduced = (mapping, parents, l_child, r_child, pruned)

        # JE TO SLOZITEJSI
        # ted na to nemam, ale v principu jde o to, invalidovat jen levou vetev, v pravy se jeste muze neco dit...
        # leaf neni jasnej, lepsi bude udelat mapovani, v pripade thresh = 0 jed doprava dokud thresh neni > 0 a pak nastav mapovani
        # jakmile dojdes na leaf, tak ho proste nastav
        # mej tam leaf priznak pouze, strukturu treba jako ma sklearn, akorat i s parent vektorem
        # pak se pro vsechny co maj rozhodnuti na 2 leaves koukej jestli nejsou obe classy stejny. Pokud ano, merge. Pak pripadne pridej dalsi
        # uzel jestli vznikl

        # return np.any(~self.__used_dec)

    def visualize_reduced(self, path, view=False, data_handler=None, show_normalized_thresholds=True):
        data_h = data_handler if data_handler is not None else self.__model_context["data_h"]
        dot = Digraph(comment="Decision tree")

        (mapping, parents, l_child, r_child, pruned) = self.__reduced

        for i, mapped in enumerate(mapping):
            if i in pruned:
                continue
            if l_child[i] != -1:
                if show_normalized_thresholds:
                    thresh = self.__model_context['b'][mapped[0]] # decisions can be mapped to a single value only
                else:
                    thresh = self.__thresholds[mapped[0]]
                dot.node(f"bra{i}", f"[{self.__decision_features[mapped[0]]}] ? {thresh:.5g}", tooltip="tmp", shape="rect")
                if i > 0:
                    edge_desc = f"<" if i == l_child[parents[i]] else f"≥"
                    dot.edge(f"bra{parents[i]}", f"bra{i}", edge_desc)
            else:
                points_total = sum(self.__accuracy_context['leaf_totals'][m - self.__n_decision_nodes + 1] for m in mapped)
                correct_total = sum(self.__accuracy_context['leaf_corr'][m - self.__n_decision_nodes + 1] for m in mapped)
                acc = correct_total / points_total if points_total > 0 else 1
                desc =  f"{points_total} ({acc*100:.3g}%)"
                c = self.__leaf_assignments[mapped[0] - self.__n_decision_nodes + 1]
                dot.node(f"dec{i}", desc, tooltip="tmp", shape="circle", color="red" if c == 1 else "green")
                edge_desc = f"<" if i == l_child[parents[i]] else f"≥"
                dot.edge(f"bra{parents[i]}", f"dec{i}", edge_desc)

        dot.format = "pdf"
        dot.render(path, view=view)

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
        self.__accuracy_context["leaf_corr"] = leaf_corr
        self.__accuracy_context["leaf_acc"] = leaf_acc
        self.__accuracy_context["leaf_totals"] = leaf_tot
        if return_computed:
            return np.nanmin(leaf_acc), tot_corr.mean()
        else:
            return leaf_acc, tot_corr

    def visualize(self, path, view=False, data_handler=None, show_normalized_thresholds=True):
        data_h = data_handler if data_handler is not None else self.__model_context["data_h"]
        dot = Digraph(comment="Decision tree")

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
                desc =  f"{self.__accuracy_context['leaf_totals'][node]} [{self.__accuracy_context['leaf_acc'][node]*100:.3g}%]"
            else:
                desc =  f"{self.__accuracy_context['leaf_totals'][node]} ({self.__accuracy_context['leaf_acc'][node]*100:.3g}%)"
            dot.node(f"dec{node}", desc, tooltip="tmp", shape="circle", color="red" if c == 1 else "green")#, style="filled")
            # dot.node(f"dec{node}", f"{data_h.class_mapping[c]}", tooltip="tmp", shape="circle", color="red" if c == 1 else "green", style="filled")

            parent_i = (node+offset) // 2
            if show_normalized_thresholds:
                thresh = self.__model_context['b'][parent_i]
            else:
                thresh = self.__thresholds[parent_i]
            edge_desc = f"< {thresh:.2f}" if node % 2 == 0 else f"≥ {thresh:.2f}"
            dot.edge(f"bra{parent_i}", f"dec{node}", edge_desc)

        dot.format = "pdf"
        dot.render(path, view=view)

    def as_ab_values(self):
        if self.__model_context is not None \
           and "a" in self.__model_context \
           and "b" in self.__model_context:
            return self.__model_context["a"], self.__model_context["b"]
        else:
            raise NotImplementedError("Case when a or b is not already present has not been considered yet.")

    def compute_similarity(self, ohter_tree):
        size = min(self.__decision_features.shape[0], ohter_tree.decision_features.shape[0])
        same = self.__decision_features[:size] == ohter_tree.decision_features[:size]
        similarity = 1 - np.abs(self.__model_context["b"][:size] - ohter_tree.as_ab_values()[1][:size])
        return (similarity * same).sum() / size

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
            return self.__accuracy_context["bellow_threshold"]

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
