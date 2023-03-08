import numpy as np
import time
import os
import pickle
import argparse

from xct_nn.XCT_MIP import XCT_MIP
from xct_nn.DataHandler import DataHandler
from xct_nn.UtilityHelper import UtilityHelper

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("-data", "--dataset_path", required=True, help="Path to the dataset")
parser.add_argument("-type", "--dataset_type", required=True, help="Either categorical or numerical")
parser.add_argument("-seed", "--random_seed", type=int, default=0, help="Number with which to seed the data split")
parser.add_argument("-r", "--round_limit", type=int, default=5, help="Max number of decimals in original data")

# output parameters
parser.add_argument("-res", "--results_dir", required=True, help="Where to store the results")
parser.add_argument("-v", "--verbose",  action="store_true", help="Print model and stats to stdout")

# model parameters
parser.add_argument("-hard", "--hard_constr", action="store_true", help="Go with hard constraint in leaves")
parser.add_argument("-init", "--init_type", default="warmstart", help="How should the values of previous tree be used? [warmstart, hint, fix_values]")
parser.add_argument("-d", "--depth", type=int, default=5, help="Final depth of the tree")
parser.add_argument("-max", "--max_data", type=int, default=10_000, help="Limit on data inputed into the model")

# optimization parameters
parser.add_argument("-t", "--time_limit", type=int, default=1800, help="Total time limit. Each depth has double of the previous time limit")
parser.add_argument("-m", "--memory_limit", type=int, default=None, help="Memory limit for gurobi [GB]")
parser.add_argument("-thr", "--n_threads", type=int, default=None, help="Number of threads for gurobi to use")
parser.add_argument("-focus", "--mip_focus", type=int, default=0, help="Value of MIPFocus parameter for Gurobi")
parser.add_argument("-heur", "--mip_heuristics", type=float, default=0.05, help="Value of Heuristics parameter for Gurobi")
args = parser.parse_args()

with open(args.dataset_path, "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {args.dataset_type}")

data_handler = DataHandler(args.dataset_path, round_limit=args.round_limit)
util = UtilityHelper(data_handler)
X_train, y_train = data_handler.get_training_data(split_seed=args.random_seed, test_size=0.2, limit=args.max_data)

logfile_base = args.results_dir + f"/run{args.random_seed}"
time_limit = args.time_limit / (2**args.depth - 1)

values = None
for depth in range(1, args.depth+1):
    print(f"Creating model with depth {depth}...")
    xct = XCT_MIP(depth, data_handler, hard_constraint=args.hard_constr)
    xct.make_model(X_train, y_train)

    print("Optimizing the model...")
    initialize = args.init_type if values is not None else None
    res = xct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, mip_focus=args.mip_focus, mip_heuristics=mip_heuristics, log_file=f"{logfile_base}_d{depth}.log", initialize=initialize, values=values, verbose=args.verbose)

    time_limit *= 2 # double the time limit after each depth
    status = xct.get_humanlike_status()

    if res:
        acc = xct.model.getObjective().getValue()

        ctx = xct.get_base_context()
        ctx["train_leaf_acc"], ctx["train_acc"] = util.get_accuracy_from_ctx(ctx, *data_handler.used_data)
        ctx["test_leaf_acc"], ctx["test_acc"] = util.get_accuracy_from_ctx(ctx, *data_handler.test_data)
        # get these stats from the re-represented tree, since it does not have the numerical issues
        ctx["n_soft_constrained"] = util.are_soft_constrained(ctx)
        ctx["n_empty_leaves"] = util.n_empty_leaves(ctx)
        problem, diff = util.check_leaf_assignment(xct)
        misassigned = np.abs(diff).sum()/2
        ctx["n_misassigned"] = misassigned
        if problem:
            print(f"Problem with solution: {misassigned} points misassigned")
            print("Differences:", diff)

        with open(f"{logfile_base}_d{depth}.ctx", "wb") as f:
            pickle.dump(ctx, f)

        # prep the values (add depth)
        # new_a = np.zeros((X_train.shape[1], 2**(depth+1)-1))
        # new_b = np.zeros((2**(depth+1)-1,))
        # new_a[:, :(2**depth - 1)] = ctx["a"]
        # new_a[0, (2**depth - 1):] = 1
        # new_b[:(2**depth - 1)] = ctx["b"]
        # values = new_a, new_b

        # adding depth is would not work with fixing the upper levels and does not help much in other cases
        values = ctx["a"], ctx["b"]

        xct.model.write(f"{logfile_base}_d{depth}.sol")
        print(f"At depth {depth} found a solution with {acc*100} leaf accuracy - {status}")
    else:
        print(f"At depth {depth} did not find a solution - {status}")
        if status == "INF":
            xct.model.computeIIS()
            xct.model.write(f"{logfile_base}_{status}.ilp")
        break
