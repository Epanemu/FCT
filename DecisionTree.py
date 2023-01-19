import gurobipy as gb
import numpy as np

class DecisionTreeMIP:
    leaf_acc_limit = 20 # since how many points compute precision
    max_invalid = 5 # require at most this many misclasified

    def __init__(self, depth, leaf_accuracy=0.9, min_in_leaf=1, only_feasibility=False):
        self.depth = depth
        self.leaf_accuracy = leaf_accuracy
        self.min_in_leaf = min_in_leaf
        self.only_feasibility = only_feasibility
        self.max_invalid = max(self.leaf_acc_limit * (1-leaf_accuracy), self.max_invalid)

    def fit_model(self, X, y, n_classes, epsilons, warmstart_values=None, time_limit=3600, verbose=False, log_file=""):
        n_data, n_features = X.shape

        leaf_nodes = 2**self.depth
        branch_nodes = 2**self.depth - 1

        left_ancestors = [] # those where decision went left
        right_ancestors = [] # those where decision went right
        for leaf_i in range(leaf_nodes):
            left_ancestors.append([])
            right_ancestors.append([])
            prev_i = leaf_i+branch_nodes
            for _ in range(self.depth):
                parent_i = (prev_i-1) // 2
                if (prev_i-1) % 2:
                    right_ancestors[leaf_i].append(parent_i)
                else:
                    left_ancestors[leaf_i].append(parent_i)
                prev_i = parent_i
        # EXAMPLE
        # node indices for self.depth = 2
        #        0
        #    1       2
        #  3   4   5   6
        #  0   1   2   3 # true indices of leaf nodes
        # print(left_ancestors) # [[1, 0], [0], [2], []]
        # print(right_ancestors) # [[], [1], [0], [2, 0]]

        # MAKE THE MILP MODEL
        m = gb.Model("DT model")

        # branch nodes computation conditions
        a = m.addMVar((n_features, branch_nodes), vtype=gb.GRB.BINARY, name="a")
        b = m.addMVar((branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b")
        # warm start
        if warmstart_values is not None:
            if verbose:
                print("warm starting the model")
            initial_a, initial_b = warmstart_values
            a.Start = initial_a
            b.Start = initial_b

        # variable d replaced with set 1
        m.addConstr(a.sum(axis=0) == 1) # (2)
        m.addConstr(b <= 1) # (3)

        # leaf nodes assignment conditions
        point_assigned = m.addMVar((n_data, leaf_nodes), vtype=gb.GRB.BINARY, name="point_assigned") # variable z
        any_assigned = m.addMVar((leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned") # variable l
        m.addConstr(point_assigned <= any_assigned) # (6)
        # if any point is assigned, the node must be assigned at least self.min_in_leaf in total
        m.addConstr(point_assigned.sum(axis=0) >= any_assigned * self.min_in_leaf) # (7)
        # points assigned to exactly one leaf
        m.addConstr(point_assigned.sum(axis=1) == 1) # (8)

        M_right = 1
        M_left = 1 + epsilons.max()
        # conditions for assignment to node
        for leaf_i in range(leaf_nodes):
            if right_ancestors[leaf_i]: # causes issues if there are no ancestors
                m.addConstr(X @ a[:, right_ancestors[leaf_i]] >= b[np.newaxis, right_ancestors[leaf_i]] - M_right*(1-point_assigned[:,[leaf_i]])) # (10)
            if left_ancestors[leaf_i]:
                m.addConstr((X + epsilons) @ a[:, left_ancestors[leaf_i]] <= b[np.newaxis, left_ancestors[leaf_i]] + M_left*(1-point_assigned[:,[leaf_i]])) # (12)

        # classification
        # Y reworked to 0 or 1
        Y = np.zeros((n_classes, n_data))
        for c in range(n_classes):
            Y[c, y == c] = 1

        class_points_in_leaf = m.addMVar((n_classes, leaf_nodes), name="N_class_points_in_leaf") # variable N_kt
        m.addConstr(class_points_in_leaf == Y @ point_assigned) # (15)

        # TODO could be ommitted, likely
        points_in_leaf = m.addMVar((leaf_nodes,), name="N_points_in_leaf") # variable N_t
        m.addConstr(points_in_leaf == point_assigned.sum(axis=0)) # (16)

        # if any nodes are assigned to leaf, it needs a class
        class_in_leaf = m.addMVar((n_classes, leaf_nodes), vtype=gb.GRB.BINARY, name="class_in_leaf") # variable c
        m.addConstr(class_in_leaf.sum(axis=0) == any_assigned) # (18)

        # accuracy measure
        M = n_data
        misclassified = m.addMVar((leaf_nodes,), lb=0, name="n_misclassfiied") # variable L (misclassification loss)
        # essentially looks for minimum of the number of misclassified samples
        m.addConstr(misclassified >= points_in_leaf - class_points_in_leaf - M * (1 - class_in_leaf)) # (20)
        m.addConstr(misclassified <= points_in_leaf - class_points_in_leaf + M * class_in_leaf) # (21)

        # ADDED:
        # Require defined accuracy in leaves
        # Either accuracy, or if not many points in leaf maximal number of misclasifications
        # SOFT CONSTRAINT
        use_acc = m.addMVar((leaf_nodes,), vtype=gb.GRB.BINARY, name="uses_accuracy")
        m.addConstr(use_acc <= points_in_leaf / self.leaf_acc_limit)
        m.addConstr(use_acc >= (points_in_leaf - self.leaf_acc_limit + 1) / n_data)
        m.addConstr(misclassified <= self.max_invalid + M * use_acc)
        m.addConstr(misclassified <= points_in_leaf * (1 - self.leaf_accuracy) + M * (1 - use_acc))

        # HARD CONSTRAINT
        # m.addConstr(misclassified <= points_in_leaf * (1 - self.leaf_accuracy))

        # normalize by the number of misclassified points, if simply the most represented class would be estimated
        base_error = n_data - Y.sum(axis=1).max()
        if self.only_feasibility:
            m.setObjective(0, sense=gb.GRB.MINIMIZE) # test feasibility
        else:
            m.setObjective(misclassified.sum() / base_error, sense=gb.GRB.MINIMIZE) # (23)

        if verbose:
            m.update()
            m.printStats()
        else:
            if log_file != "":
                m.setParam('LogFile', log_file)
                m.setParam('LogToConsole', 0)
            else:
                m.setParam('OutputFlag', 0)
        # m.display()
        m.setParam('TimeLimit', time_limit)

        m.optimize()

        # TODO improve this so it does not need to return it, model stays in between runs
        return m.SolCount > 0, m, a, b
        # if m.status == gb.GRB.TIME_LIMIT:
        #     obj = m.getObjective()
        #     return obj.getValue()
        # return m.status != gb.GRB.INFEASIBLE
