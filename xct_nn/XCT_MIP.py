
try:
    import gurobipy as gb
except ImportError:
    print("Gurobi is not available, related functions will fail.")
import numpy as np

class XCT_MIP:
    def __init__(self, depth, data_handler, min_in_leaf=1, leaf_accuracy=None, leaf_acc_limit=20, max_invalid=None, only_feasibility=False, hard_constraint=False):
        self.depth = depth
        self.data_h = data_handler
        self.min_in_leaf = min_in_leaf
        self.leaf_accuracy = leaf_accuracy
        self.maximize_leaf_accuracy = leaf_accuracy is None
        self.leaf_acc_limit = leaf_acc_limit
        self.max_invalid = max_invalid
        if max_invalid is None and leaf_accuracy is not None:
            self.max_invalid = leaf_acc_limit * (1-leaf_accuracy)
        self.only_feasibility = only_feasibility
        self.hard_constraint = hard_constraint

        self.__n_leaf_nodes = 2**self.depth
        self.__n_branch_nodes = 2**self.depth - 1
        self.model = None

    def make_model(self, X, y):
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
        self.model = gb.Model("FCT model")

        # NOTE the constraint numbers in parenthesses refer to the numbers of constraints in the OCT formulation paper

        # branch nodes computation conditions
        a = self.model.addMVar((self.data_h.n_features, self.__n_branch_nodes), vtype=gb.GRB.BINARY, name="a")
        b = self.model.addMVar((self.__n_branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b")

        # variable d replaced with set 1
        self.model.addConstr(a.sum(axis=0) == 1) # (2)
        # self.model.addConstr(b <= 1) # (3)

        # leaf nodes assignment conditions
        point_assigned = self.model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="point_assigned") # variable z
        any_assigned = self.model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned") # variable l
        self.model.addConstr(point_assigned <= any_assigned) # (6)
        # if any point is assigned, the node must be assigned at least self.min_in_leaf in total
        self.model.addConstr(point_assigned.sum(axis=0) >= any_assigned * self.min_in_leaf) # (7)
        # points assigned to exactly one leaf
        self.model.addConstr(point_assigned.sum(axis=1) == 1) # (8)

        M_right = 1
        M_left = 1 + self.data_h.epsilons.max()
        # conditions for assignment to node
        for leaf_i in range(self.__n_leaf_nodes):
            if right_ancestors[leaf_i]: # causes issues if there are no ancestors
                self.model.addConstr(X @ a[:, right_ancestors[leaf_i]] >= b[np.newaxis, right_ancestors[leaf_i]] - M_right*(1-point_assigned[:,[leaf_i]])) # (10)
            if left_ancestors[leaf_i]:
                self.model.addConstr((X + self.data_h.epsilons) @ a[:, left_ancestors[leaf_i]] <= b[np.newaxis, left_ancestors[leaf_i]] + M_left*(1-point_assigned[:,[leaf_i]])) # (12)

        # classification
        # Y reworked to 0 or 1
        Y = np.zeros((self.data_h.n_classes, self.data_h.n_data))
        for c in range(self.data_h.n_classes):
            Y[c, y == c] = 1

        class_in_leaf = self.model.addMVar((self.data_h.n_classes, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="class_in_leaf") # variable c
        # if any nodes are assigned to leaf, it needs a class
        self.model.addConstr(class_in_leaf.sum(axis=0) == any_assigned) # (18)

        self.vars = {
            "a": a,
            "b": b,
            "point_assigned": point_assigned,
            "any_assigned": any_assigned,
            "class_in_leaf": class_in_leaf,
        }

        if not self.hard_constraint or not self.maximize_leaf_accuracy:
            class_points_in_leaf = self.model.addMVar((self.data_h.n_classes, self.__n_leaf_nodes), name="N_class_points_in_leaf") # variable N_kt
            self.model.addConstr(class_points_in_leaf == Y @ point_assigned) # (15)
            self.vars["class_points_in_leaf"] = class_points_in_leaf

            # extra variable could be ommitted. It is here to provide simpler information and less occlusion in other constraints
            points_in_leaf = self.model.addMVar((self.__n_leaf_nodes,), name="N_points_in_leaf") # variable N_t
            self.model.addConstr(points_in_leaf == point_assigned.sum(axis=0)) # (16)
            self.vars["points_in_leaf"] = points_in_leaf

        if not self.hard_constraint:
            # must add variable signifying what constraint is used
            is_hard = self.model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="is_hard_constrained") # variable h
            self.vars["is_hard"] = is_hard
            self.model.addConstr(is_hard <= points_in_leaf / self.leaf_acc_limit)
            self.model.addConstr(is_hard >= (points_in_leaf - self.leaf_acc_limit + 1) / self.data_h.n_data)
        M = self.data_h.n_data

        if self.maximize_leaf_accuracy:
            # how much accuracy can a datapoint in a leaf give
            accuracy_ammount = self.model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), lb=0, ub=1, name="accuracy_ammount") # variable s
            self.vars["accuracy_ammount"] = accuracy_ammount
            self.model.addConstr(accuracy_ammount.sum(axis=0) == any_assigned)  # ideal is 100% acuracy

            accuracy_ref = self.model.addMVar((self.__n_leaf_nodes,), lb=0, ub=1, name="accuracy_reference") # variable r
            self.vars["accuracy_reference"] = accuracy_ref
            # accuracy equal to 0 if not assigned
            self.model.addConstr(accuracy_ammount <= point_assigned)
            # or to reference if assigned
            self.model.addConstr(accuracy_ref <= accuracy_ammount + (1 - point_assigned))
            self.model.addConstr(accuracy_ref >= accuracy_ammount + (point_assigned - 1))

            # how much accuracy each datapoint in a leaf gives
            assigned_accuracy = self.model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), lb=0, ub=1, name="assigned_accuracy") # variable S
            self.vars["assigned_accuracy"] = assigned_accuracy
            self.model.addConstr(assigned_accuracy <= accuracy_ammount) # accuracy can be either equal to its potential or to 0
            self.model.addConstr(assigned_accuracy >= accuracy_ammount + (Y.T @ class_in_leaf - 1)) # force it equal to potential
            self.model.addConstr(assigned_accuracy <= Y.T @ class_in_leaf) # force it to 0, if misclassified

            leaf_acc = self.model.addVar(lb=0, ub=1, name="leaf_accuracy") # variable Q
            self.vars["leaf_acc"] = leaf_acc
            if self.hard_constraint:
                self.model.addConstr(leaf_acc <= assigned_accuracy.sum(axis=0) + (1 - any_assigned)) # lowest bound on leaf accuracy
            else:
                # original accuracy measure
                misclassified = self.model.addMVar((self.__n_leaf_nodes,), lb=0, name="n_misclassfiied") # variable L (misclassification loss)
                self.vars["misclassified"] = misclassified
                # essentially looks for minimum of the number of misclassified samples
                self.model.addConstr(misclassified >= points_in_leaf - class_points_in_leaf - M * (1 - class_in_leaf)) # (20)
                self.model.addConstr(misclassified <= points_in_leaf - class_points_in_leaf + M * class_in_leaf) # (21)

                if self.max_invalid is None:
                    # If max invalid is not set directly, change the limit according to the current leaf accuracy
                    self.model.addConstr(misclassified <= self.leaf_acc_limit * (1 - leaf_acc) + M * is_hard)
                else:
                    # fixed maximum of invalid points
                    self.model.addConstr(misclassified <= self.max_invalid + M * is_hard)
                # hard constraint as in the hard constrained variant
                self.model.addConstr(leaf_acc <= assigned_accuracy.sum(axis=0) + (1 - any_assigned)  + (1 - is_hard))

            self.model.setObjective(leaf_acc, sense=gb.GRB.MAXIMIZE)
        else:
            # original accuracy measure
            misclassified = self.model.addMVar((self.__n_leaf_nodes,), lb=0, name="n_misclassfiied") # variable L (misclassification loss)
            self.vars["misclassified"] = misclassified
            # essentially looks for minimum of the number of misclassified samples
            self.model.addConstr(misclassified >= points_in_leaf - class_points_in_leaf - M * (1 - class_in_leaf)) # (20)
            self.model.addConstr(misclassified <= points_in_leaf - class_points_in_leaf + M * class_in_leaf) # (21)

            # ADDED:
            # Require defined accuracy in leaves
            # Either accuracy, or if not many points in leaf maximal number of misclasifications
            if self.hard_constraint:
                # HARD CONSTRAINT
                self.model.addConstr(misclassified <= points_in_leaf * (1 - self.leaf_accuracy))
            else:
                # SOFT CONSTRAINT
                self.model.addConstr(misclassified <= self.max_invalid + M * is_hard)
                self.model.addConstr(misclassified <= points_in_leaf * (1 - self.leaf_accuracy) + M * (1 - is_hard))


            # normalize by the number of misclassified points, if simply the most represented class would be estimated
            base_error = self.data_h.n_data - Y.sum(axis=1).max()
            if self.only_feasibility:
                self.model.setObjective(0, sense=gb.GRB.MINIMIZE) # test feasibility
            else:
                self.model.setObjective(misclassified.sum() / base_error, sense=gb.GRB.MINIMIZE) # (23)

        self.model.update()

    def optimize(self, initialize=None, values=None, time_limit=3600, mem_limit=None, n_threads=None, mip_focus=0, mip_heuristics=0.05, verbose=False, log_file=""):
        assert self.model is not None

        # initialize the tree values
        if initialize is not None:
            if initialize == "warmstart":
                if values is None:
                    raise ValueError("Must provide values if warmstarting")
                if verbose:
                    print("Warm-starting the model")
                initial_a, initial_b = values
                for i in range(initial_b.shape[0]): # in case I do not wish to warmstart the entire tree
                    self.vars["a"][:, i].Start = initial_a[:, i].round(0)
                    self.vars["b"][i].Start = initial_b[i]
            elif initialize == "hint":
                if values is None:
                    raise ValueError("Must provide values if hinting")
                if verbose:
                    print("Hinting some values of the model")
                initial_a, initial_b = values
                for i in range(initial_b.shape[0]):
                    self.vars["a"][:, i].VarHintVal = initial_a[:, i].round(0)
                    self.vars["b"][i].VarHintVal = initial_b[i]
                    self.vars["a"][:, i].Start = initial_a[:, i].round(0)
                    self.vars["b"][i].Start = initial_b[i]
            elif initialize == "fix_values":
                if values is None:
                    raise ValueError("Must provide values if fixing the upper tree")
                if verbose:
                    print("Fixing some decisions in the model")
                initial_a, initial_b = values
                for i in range(initial_b.shape[0]):
                    # can lead to infeasible models if the values have are not numerically stable
                    self.model.addConstr(self.vars["a"][:, i] == initial_a[:, i].round(0))
                    self.model.addConstr(self.vars["b"][i] == initial_b[i].round(self.data_h.round_limit))
            else:
                raise ValueError("Unsupported value of the `initialize` parameter:", initialize)

        if verbose:
            self.model.update()
            self.model.printStats()
            self.model.display()
        else:
            if log_file != "":
                self.model.params.LogFile = log_file
                self.model.params.LogToConsole = 0
            else:
                self.model.params.OutputFlag = 0
        self.model.params.TimeLimit = time_limit
        if mem_limit is not None:
            self.model.params.SoftMemLimit = mem_limit
        self.model.params.NodefileStart = 0.5
        self.model.params.NodefileDir = "nodefiles"
        self.model.params.MIPFocus = mip_focus
        self.model.params.Heuristics = mip_heuristics
        if n_threads is not None:
            self.model.params.Threads = n_threads

        self.model.optimize()

        return self.model.SolCount > 0 # return whether a solution was found

    def get_humanlike_status(self):
        if self.model.Status == gb.GRB.OPTIMAL:
            return "OPT"
        elif self.model.Status == gb.GRB.INFEASIBLE:
            return "INF"
        elif self.model.Status == gb.GRB.TIME_LIMIT:
            return "TIME"
        elif self.model.Status == gb.GRB.MEM_LIMIT:
            return "MEM"
        elif self.model.Status == gb.GRB.INTERRUPTED:
            return "INT"
        else:
            return f"ST{self.model.status}"

    def get_base_context(self):
        return {
            "depth": self.depth,
            "data_h_setup": self.data_h.get_setup(),
            "n_data": self.data_h.n_data,
            "n_features": self.data_h.n_features,
            "min_in_leaf": self.min_in_leaf,
            "leaf_accuracy": self.leaf_accuracy,
            "maximize_leaf_accuracy": self.maximize_leaf_accuracy,
            "leaf_acc_limit": self.leaf_acc_limit,
            "max_invalid": self.max_invalid,
            "only_feasibility": self.only_feasibility,
            "hard_constraint": self.hard_constraint,
            "a": self.vars["a"].X,
            "b": self.vars["b"].X,
            "classes": self.vars["class_in_leaf"].X,
            "objective_bound": self.model.ObjBound if self.model is not None else None,
            "objective_gap": self.model.MIPGap if self.model is not None else None,
            "status": self.get_humanlike_status(),
        }

    def load_sol(self, sol_file):
        self.__dummy_model = gb.Model()
        self.__dummy_model.params.OutputFlag = 0

        a = self.__dummy_model.addMVar((self.data_h.n_features, self.__n_branch_nodes), vtype=gb.GRB.BINARY, name="a")
        b = self.__dummy_model.addMVar((self.__n_branch_nodes,), lb=0, ub=1, vtype=gb.GRB.CONTINUOUS, name="b")
        point_assigned = self.__dummy_model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="point_assigned") # variable z
        any_assigned = self.__dummy_model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="any_assigned") # variable l
        class_points_in_leaf = self.__dummy_model.addMVar((self.data_h.n_classes, self.__n_leaf_nodes), name="N_class_points_in_leaf") # variable N_kt
        points_in_leaf = self.__dummy_model.addMVar((self.__n_leaf_nodes,), name="N_points_in_leaf") # variable N_t
        class_in_leaf = self.__dummy_model.addMVar((self.data_h.n_classes, self.__n_leaf_nodes), vtype=gb.GRB.BINARY, name="class_in_leaf") # variable c

        self.vars = {
            "a": a,
            "b": b,
            "point_assigned": point_assigned,
            "any_assigned": any_assigned,
            "class_in_leaf": class_in_leaf,
        }

        if not self.hard_constraint or not self.maximize_leaf_accuracy:
            self.vars["points_in_leaf"] = points_in_leaf
            self.vars["class_points_in_leaf"] = class_points_in_leaf

        if not self.hard_constraint:
            is_hard = self.__dummy_model.addMVar((self.__n_leaf_nodes,), vtype=gb.GRB.BINARY, name="is_hard_constrained")
            self.vars["is_hard"] = is_hard

        if self.maximize_leaf_accuracy:
            accuracy_ammount = self.__dummy_model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), lb=0, ub=1, name="accuracy_ammount")
            assigned_accuracy = self.__dummy_model.addMVar((self.data_h.n_data, self.__n_leaf_nodes), lb=0, ub=1, name="assigned_accuracy")
            leaf_acc = self.__dummy_model.addVar(lb=0, ub=1, name="leaf_accuracy")
            self.vars["accuracy_ammount"] = accuracy_ammount
            self.vars["assigned_accuracy"] = assigned_accuracy
            self.vars["leaf_acc"] = leaf_acc
        else:
            misclassified = self.__dummy_model.addMVar((self.__n_leaf_nodes,), lb=0, name="n_misclassfiied") # variable L (misclassification loss)
            self.vars["misclassified"] = misclassified

        self.__dummy_model.update()
        self.__dummy_model.read(sol_file)
        self.__dummy_model.optimize()

        self.model = None # should not optimize after this, need to rebuild the model
