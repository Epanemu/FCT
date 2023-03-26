import numpy as np
from .ClassificationTree import ClassificationTree
from .XCT_MIP import XCT_MIP

class TreeGenerator:
    def __init__(self, data_handler):
        self.data_h = data_handler

    def make_from_context(self, context):
        decision_features = context["a"].argmax(axis=0)
        # thresholds = context["b"].round(self.data_h.round_limit) * self.data_h.scales[decision_features] + self.data_h.shifts[decision_features]
        thresholds = self.__thresholds_from_b(context["b"], decision_features)
        leaf_assignments = context["classes"].argmax(axis=0)

        return ClassificationTree(context, decision_features, thresholds, leaf_assignments)

    def make_from_matrices(self, a_matrix, b_matrix, classes_matrix, depth=None):
        if depth is None:
            depth = self.__depth_from_branchnodes(b_matrix.shape[0])

        decision_features = a_matrix.argmax(axis=0)
        # thresholds = b_matrix.round(self.data_h.round_limit) * self.data_h.scales[decision_features] + self.data_h.shifts[decision_features]
        thresholds = self.__thresholds_from_b(b_matrix, decision_features)
        leaf_assignments = classes_matrix.argmax(axis=0)

        base_context = {
            "depth": depth,
            "a": a_matrix.round(0),
            "b": b_matrix,
            "classes": classes_matrix.round(0),
        }

        return ClassificationTree(base_context, decision_features, thresholds, leaf_assignments)

    def make_from_SOL_file(self, sol_file, depth=None):
        if depth is None:
            with open(sol_file) as f:
                branch_nodes = len([line for line in f if "b[" == line[:2]])
                depth = self.__depth_from_branchnodes(branch_nodes)
        mip_model = XCT_MIP(depth, self.data_h)
        mip_model.load_sol(sol_file)
        return self.make_from_matrices(mip_model.vars["a"].X, mip_model.vars["b"].X, mip_model.vars["class_in_leaf"].X, depth)

    def __depth_from_branchnodes(self, branch_nodes):
        return np.log2(branch_nodes + 1).astype(int)

    def __thresholds_from_b(self, b, dec_features):
        return np.clip(b, 0, 1).round(self.data_h.round_limit) * self.data_h.scales[dec_features] + self.data_h.shifts[dec_features]

    def __depth_from_index(self, i):
        return np.floor(np.log2(i + 1)).astype(int)

    def make_from_sklearn(self, sklearn_tree, soft_limit, train_data, normalized_thresholds=True):
        epsilons = self.data_h.epsilons
        if not normalized_thresholds:
            epsilons = self.data_h.unnormalize(epsilons)

        totdepth = sklearn_tree.max_depth
        decision_features = np.zeros((2**totdepth - 1,), dtype=int)
        thresholds = np.zeros((2**totdepth - 1,))
        leaf_assignments = np.zeros((2**totdepth,), dtype=int)
        map_i = np.full((len(sklearn_tree.children_left),), -1)
        queue = [(0, 0, 0)]
        while queue:
            i, depth, mapped = queue.pop(0)
            map_i[i] = mapped
            if sklearn_tree.children_left[i] != -1: # if node is not a leaf
                queue.append((sklearn_tree.children_left[i], depth+1, mapped*2+1))
                queue.append((sklearn_tree.children_right[i], depth+1, mapped*2+2))
                dec_feat_i = sklearn_tree.feature[i]
                decision_features[mapped] = dec_feat_i

                # sklearn uses <=, this implementation uses <
                vals = train_data[:, dec_feat_i]
                orig_thresh = sklearn_tree.threshold[i]
                lower_equal_data = vals <= orig_thresh
                diff = orig_thresh - vals[lower_equal_data]
                closest_i = np.argmin(diff)
                if diff[closest_i] < epsilons[dec_feat_i]:
                    thresholds[mapped] = vals[lower_equal_data][closest_i] + epsilons[dec_feat_i]
                else:
                    thresholds[mapped] = orig_thresh
                # or this, although this leads to more and bigger changes to the thresholds
                # thresholds[mapped] = np.min(vals[vals > orig_thresh])
                if not normalized_thresholds:
                    thresholds[mapped] = (thresholds[mapped] - self.data_h.shifts[dec_feat_i]) / self.data_h.scales[dec_feat_i]
            else:
                depth_remaining = totdepth - self.__depth_from_index(mapped)
                for _ in range(depth_remaining):
                    mapped = mapped*2 + 2
                leaf_assignments[mapped - 2**totdepth + 1] = np.argmax(sklearn_tree.value[i])

        a_matrix = np.zeros((self.data_h.n_features, decision_features.shape[0]), dtype=int)
        a_matrix[decision_features, np.arange(decision_features.shape[0])] = 1
        classes_matrix = np.zeros((self.data_h.n_classes, leaf_assignments.shape[0]), dtype=int)
        classes_matrix[leaf_assignments, np.arange(leaf_assignments.shape[0])] = 1
        base_context = {
            "data_h": self.data_h,
            "depth": totdepth,
            "a": a_matrix,
            "b": thresholds,
            "classes": classes_matrix,
            "hard_constraint": soft_limit>0,
            "max_invalid": None,
            "leaf_acc_limit": soft_limit,
        }
        return ClassificationTree(base_context, decision_features, self.__thresholds_from_b(thresholds, decision_features), leaf_assignments)
