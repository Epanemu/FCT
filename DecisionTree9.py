import gurobipy as gb
import numpy as np

class DecisionTreeMIP:
    leaf_acc_limit = 20 # since how many points compute precision
    max_invalid = 5 # require at most this many misclasified

    def __init__(self, depth, leaf_accuracy=0.9, min_in_leaf=1):
        self.depth = depth
        self.leaf_accuracy = leaf_accuracy
        self.min_in_leaf = min_in_leaf
        self.max_invalid = max(self.leaf_acc_limit * (1-leaf_accuracy), self.max_invalid)

    def fit_model(self, X, y, n_classes, epsilons, warmstart_values=None, time_limit=3600, verbose=False):
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

        does_split = m.addMVar((branch_nodes,), vtype=gb.GRB.BINARY, name="does_split") # variable d
        m.addConstr(does_split == 1) # Added based on an another implementation
        for i in range(branch_nodes):
            m.addConstr(a[:, i].sum() == does_split[i]) # (2)
        m.addConstr(b <= does_split) # (3)

        # if a parent does not split child will not either
        for d_i in range(1, branch_nodes): # not for root
            parent_i = (d_i-1) // 2
            m.addConstr(does_split[d_i] <= does_split[parent_i]) # (5)


        # leaf nodes assignment conditions
        point_assigned = m.addMVar((n_data, leaf_nodes), vtype=gb.GRB.BINARY, name="point_assigned") # variable z
        any_assigned = m.addMVar((leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned") # variable l
        # m.addConstr(any_assigned == 1) # also added
        for i in range(leaf_nodes):
            for j in range(n_data):
                m.addConstr(point_assigned[j, i] <= any_assigned[i]) # (6)
            # if any point is assigned, the node must be assigned at least N_min in total
            m.addConstr(point_assigned[:, i].sum() >= any_assigned[i] * self.min_in_leaf) # (7)
            # points assigned to exactly one leaf
        for j in range(n_data):
            m.addConstr(point_assigned[j, :].sum() == 1) # (8)

        M_right = 1
        M_left = 1 + epsilons.max()
        # M_left = 1 + max(epsilons.max(), 0.0005)
        # conditions for assignment to node
        for leaf_i in range(leaf_nodes):
            for data_i in range(n_data):
                for ancestor_i in right_ancestors[leaf_i]: # causes issues if there are no ancestors
                    m.addConstr(X[data_i] @ a[:, ancestor_i] >= b[ancestor_i] - M_right*(1-point_assigned[data_i, leaf_i])) # (10)
                for ancestor_i in left_ancestors[leaf_i]:
                    m.addConstr((X + epsilons)[data_i,:] @ a[:, ancestor_i] <= b[ancestor_i] + M_left*(1-point_assigned[data_i, leaf_i])) # (12)

        # classification
        Y = np.zeros((n_classes, n_data))
        for c in range(n_classes):
            Y[c, y == c] = 1

        class_points_in_leaf = m.addMVar((n_classes, leaf_nodes), name="N_class_points_in_leaf") # variable N_kt
        for leaf_i in range(leaf_nodes):
            for class_i in range(n_classes):
                m.addConstr((Y[class_i, :] @ point_assigned[:, leaf_i] == class_points_in_leaf[class_i, leaf_i])) # (15)

        # TODO could be ommitted, likely
        points_in_leaf = m.addMVar((leaf_nodes,), name="N_points_in_leaf") # variable N_t
        for leaf_i in range(leaf_nodes):
            m.addConstr((point_assigned[:, leaf_i].sum() == points_in_leaf[leaf_i])) # (16)

        # if any nodes are assigned to leaf, it needs a class
        class_in_leaf = m.addMVar((n_classes, leaf_nodes), vtype=gb.GRB.BINARY, name="class_in_leaf") # variable c

        for leaf_i in range(leaf_nodes):
            m.addConstr(class_in_leaf[:, leaf_i].sum() == any_assigned[leaf_i]) # (18)

        # accuracy measure
        M = n_data
        misclassified = m.addMVar((leaf_nodes,), lb=0, name="n_misclassfiied") # variable L (misclassification loss)
        # essentially looks for minimum of the number of misclassified samples
        for leaf_i in range(leaf_nodes):
            for class_i in range(n_classes):
                m.addConstr(misclassified[leaf_i] >= points_in_leaf[leaf_i] - class_points_in_leaf[class_i, leaf_i] - M * (1 - class_in_leaf[class_i, leaf_i])) # (20)
                m.addConstr(misclassified[leaf_i] <= points_in_leaf[leaf_i] - class_points_in_leaf[class_i, leaf_i] + M * class_in_leaf[class_i, leaf_i]) # (21)


        # ADDED:
        # Require defined accuracy in leaves
        # Either accuracy, or if not many points in leaf maximal number of misclasifications
        use_acc = m.addMVar((leaf_nodes,), vtype=gb.GRB.BINARY, name="uses_accuracy")
        for leaf_i in range(leaf_nodes):
            m.addConstr(use_acc[leaf_i] <= points_in_leaf[leaf_i]/self.leaf_acc_limit)
            m.addConstr(use_acc[leaf_i] >= (points_in_leaf[leaf_i]-self.leaf_acc_limit+1) / n_data)
            m.addConstr(misclassified[leaf_i] <= self.max_invalid + M * use_acc[leaf_i])
            m.addConstr(misclassified[leaf_i] <= points_in_leaf[leaf_i] * (1 - self.leaf_accuracy) + M * (1 - use_acc[leaf_i])) # each leave precision


        # normalize by the number of misclassified, if the most represented class would be estimated
        base_error = n_data - Y.sum(axis=1).max()
        alpha = 0 # compexity parameter
        m.setObjective(misclassified.sum() / base_error + alpha * does_split.sum(), sense=gb.GRB.MINIMIZE) # (23)

        if verbose:
            m.update()
            m.printStats()
        else:
            m.setParam('OutputFlag', 0)
        # m.display()
        m.setParam('TimeLimit', time_limit)

        m.optimize()

        # TODO improve this so it returns more than if feasible solution was found
        return m.SolCount > 0
        # if m.status == gb.GRB.TIME_LIMIT:
        #     obj = m.getObjective()
        #     return obj.getValue()
        # return m.status != gb.GRB.INFEASIBLE
