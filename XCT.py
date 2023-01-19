import gurobipy as gb
import numpy as np

class XCT_MIP:
    def __init__(self, depth, leaf_accuracy=0.9, min_in_leaf=1, leaf_acc_limit=20, max_invalid=None, only_feasibility=False, hard_constraint=False):
        self.depth = depth
        self.leaf_accuracy = leaf_accuracy
        self.min_in_leaf = min_in_leaf
        self.only_feasibility = only_feasibility
        self.hard_constraint = hard_constraint
        self.leaf_acc_limit = leaf_acc_limit
        self.max_invalid = leaf_acc_limit * (1-leaf_accuracy) if max_invalid is not None else max_invalid
        self.model = None

    def make_model(self, X, y, n_classes):
        self.__n_classes = n_classes
        self.__n_data, self.__n_features = X.shape

        self.shifts = X.min(axis=0)
        X -= self.shifts
        self.scales = X.max(axis=0)
        self.scales[self.scales == 0] = 1
        X /= self.scales

        self.epsilons = np.empty((self.__n_features,))
        for i, col_data in enumerate(X.T):
            col_sorted = col_data.copy()
            col_sorted.sort()
            eps = col_sorted[1:] - col_sorted[:-1]
            eps[eps == 0] = np.inf
            self.epsilons[i] = eps.min()

        self.epsilons[self.epsilons == np.inf] = 1 # if all values were same, we actually want eps nonzero to prevent false splitting

        self.__n_leaf_nodes = 2**self.depth
        self.__n_branch_nodes = 2**self.depth - 1

        left_ancestors = [] # those where decision went left
        right_ancestors = [] # those where decision went right
        for leaf_i in range(self.__n_leaf_nodes):
            left_ancestors.append([])
            right_ancestors.append([])
            prev_i = leaf_i + self.__n_branch_nodes
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
        self.model = gb.Model("XCT model")

        # branch nodes computation conditions
        a = self.model.addMVar((self.__n_features, self.__n_branch_nodes), vtype=gb.GRB.BINARY, name="a")
        b = self.model.addMVar((self.__n_branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b")

        # variable d replaced with set 1
        self.model.addConstr(a.sum(axis=0) == 1) # (2)
        self.model.addConstr(b <= 1) # (3)

        # leaf nodes assignment conditions
        point_assigned = self.model.addMVar((self.__n_data, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="point_assigned") # variable z
        any_assigned = self.model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned") # variable l
        self.model.addConstr(point_assigned <= any_assigned) # (6)
        # if any point is assigned, the node must be assigned at least self.min_in_leaf in total
        self.model.addConstr(point_assigned.sum(axis=0) >= any_assigned * self.min_in_leaf) # (7)
        # points assigned to exactly one leaf
        self.model.addConstr(point_assigned.sum(axis=1) == 1) # (8)

        M_right = 1
        M_left = 1 + epsilons.max()
        # conditions for assignment to node
        for leaf_i in range(self.__n_leaf_nodes):
            if right_ancestors[leaf_i]: # causes issues if there are no ancestors
                self.model.addConstr(X @ a[:, right_ancestors[leaf_i]] >= b[np.newaxis, right_ancestors[leaf_i]] - M_right*(1-point_assigned[:,[leaf_i]])) # (10)
            if left_ancestors[leaf_i]:
                self.model.addConstr((X + epsilons) @ a[:, left_ancestors[leaf_i]] <= b[np.newaxis, left_ancestors[leaf_i]] + M_left*(1-point_assigned[:,[leaf_i]])) # (12)

        # classification
        # Y reworked to 0 or 1
        Y = np.zeros((self.__n_classes, self.__n_data))
        for c in range(self.__n_classes):
            Y[c, y == c] = 1

        class_points_in_leaf = self.model.addMVar((self.__n_classes, self.__n_leaf_nodes), name="N_class_points_in_leaf") # variable N_kt
        self.model.addConstr(class_points_in_leaf == Y @ point_assigned) # (15)

        # TODO could be ommitted, likely
        points_in_leaf = self.model.addMVar((self.__n_leaf_nodes,), name="N_points_in_leaf") # variable N_t
        self.model.addConstr(points_in_leaf == point_assigned.sum(axis=0)) # (16)

        # if any nodes are assigned to leaf, it needs a class
        class_in_leaf = self.model.addMVar((self.__n_classes, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="class_in_leaf") # variable c
        self.model.addConstr(class_in_leaf.sum(axis=0) == any_assigned) # (18)

        # accuracy measure
        M = self.__n_data
        misclassified = self.model.addMVar((self.__n_leaf_nodes,), lb=0, name="n_misclassfiied") # variable L (misclassification loss)
        # essentially looks for minimum of the number of misclassified samples
        self.model.addConstr(misclassified >= points_in_leaf - class_points_in_leaf - M * (1 - class_in_leaf)) # (20)
        self.model.addConstr(misclassified <= points_in_leaf - class_points_in_leaf + M * class_in_leaf) # (21)

        self.vars = {
            "a": a,
            "b": b,
            "point_assigned": point_assigned,
            "any_assigned": any_assigned,
            "point_in_leaf": point_in_leaf,
            "class_point_in_leaf": class_point_in_leaf,
            "class_in_leaf": class_in_leaf,
            "misclassified": misclassified
        }

        # ADDED:
        # Require defined accuracy in leaves
        # Either accuracy, or if not many points in leaf maximal number of misclasifications
        if self.hard_constraint:
            # HARD CONSTRAINT
            self.model.addConstr(misclassified <= points_in_leaf * (1 - self.leaf_accuracy))
        else:
            # SOFT CONSTRAINT
            use_acc = self.model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="uses_accuracy")
            self.vars["use_acc"] = use_acc
            self.model.addConstr(use_acc <= points_in_leaf / self.leaf_acc_limit)
            self.model.addConstr(use_acc >= (points_in_leaf - self.leaf_acc_limit + 1) / self.__n_data)
            self.model.addConstr(misclassified <= self.max_invalid + M * use_acc)
            self.model.addConstr(misclassified <= points_in_leaf * (1 - self.leaf_accuracy) + M * (1 - use_acc))


        # normalize by the number of misclassified points, if simply the most represented class would be estimated
        base_error = self.__n_data - Y.sum(axis=1).max()
        if self.only_feasibility:
            self.model.setObjective(0, sense=gb.GRB.MINIMIZE) # test feasibility
        else:
            self.model.setObjective(misclassified.sum() / base_error, sense=gb.GRB.MINIMIZE) # (23)

        self.model.update()

    def optimize(self, warmstart_values=None, time_limit=3600, verbose=False, log_file=""):
        assert self.model is not None

        # warm start
        if warmstart_values is not None:
            if verbose:
                print("warm starting the model")
            initial_a, initial_b = warmstart_values
            self.vars["a"].Start = initial_a
            self.vars["b"].Start = initial_b

        if verbose:
            self.model.update()
            self.model.printStats()
            # self.model.display()
        else:
            if log_file != "":
                self.model.params.LogFile = log_file
                self.model.params.LogToConsole = 0
            else:
                self.model.params.OutputFlag = 0
        self.model.params.TimeLimit = time_limit

        self.model.optimize()

        return self.model.SolCount > 0 # return whether a solution was found

    def get_base_context(self):
        return self.vars["a"].X, self.vars["b"].X, self.shifts, self.scales, self.depth

    def load_sol(self, sol_file):
        tmp_model = gb.Model()
        tmp_model.params.OutputFlag = 0

        a = tmp_model.addMVar((self.__n_features, self.__n_branch_nodes), vtype=gb.GRB.BINARY, name="a")
        b = tmp_model.addMVar((self.__n_branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b")
        point_assigned = tmp_model.addMVar((self.__n_data, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="point_assigned") # variable z
        any_assigned = tmp_model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned") # variable l
        class_points_in_leaf = tmp_model.addMVar((self.__n_classes, self.__n_leaf_nodes), name="N_class_points_in_leaf") # variable N_kt
        points_in_leaf = tmp_model.addMVar((self.__n_leaf_nodes,), name="N_points_in_leaf") # variable N_t
        class_in_leaf = tmp_model.addMVar((self.__n_classes, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="class_in_leaf") # variable c
        misclassified = tmp_model.addMVar((self.__n_leaf_nodes,), lb=0, name="n_misclassfiied") # variable L (misclassification loss)
        use_acc = tmp_model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="uses_accuracy")

        self.vars = {
            "a": a,
            "b": b,
            "point_assigned": point_assigned,
            "any_assigned": any_assigned,
            "point_in_leaf": point_in_leaf,
            "class_point_in_leaf": class_point_in_leaf,
            "class_in_leaf": class_in_leaf,
            "misclassified": misclassified,
            "use_acc": use_acc,
        }

        tmp_model.update()
        tmp_model.read(sol_file)
        tmp_model.optimize()

        self.model = None # should not optimize after this, need to rebuild the model