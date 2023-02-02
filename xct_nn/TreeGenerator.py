import numpy as np
from .ClassificationTree import ClassificationTree
from .XCT_MIP import XCT_MIP

class TreeGenerator:
    def __init__(self, data_handler):
        self.data_h = data_handler

    def make_from_context(self, context):
        decision_features = context["a"].argmax(axis=0)
        thresholds = context["b"] * context["data_h"].scales[decision_features] + context["data_h"].shifts[decision_features]
        leaf_assignments = context["classes"].argmax(axis=0)

        return ClassificationTree(context, decision_features, thresholds, leaf_assignments)

    def make_from_matrices(self, a_matrix, b_matrix, classes_matrix, depth=None):
        if depth is None:
            depth = self.__depth_from_branchnodes(b_matrix.shape[0])

        decision_features = a_matrix.argmax(axis=0)
        thresholds = b_matrix * self.data_h.scales[decision_features] + self.data_h.shifts[decision_features]
        leaf_assignments = classes_matrix.argmax(axis=0)

        base_context = {
            "depth": depth,
            "a": a_matrix,
            "b": b_matrix,
            "classes": classes_matrix,
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
