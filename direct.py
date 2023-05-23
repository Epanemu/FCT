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
parser.add_argument("-type", "--dataset_type", default="", help="Either categorical or numerical")
parser.add_argument("-seed", "--random_seed", type=int, default=0, help="Number with which to seed the data split")
parser.add_argument("-r", "--round_limit", type=int, default=4, help="Max number of decimals in original data")

# output parameters
parser.add_argument("-res", "--results_dir", required=True, help="Where to store the results")
parser.add_argument("-v", "--verbose",  action="store_true", help="Print model and stats to stdout")

# model parameters
parser.add_argument("-soft", "--soft_constr", action="store_true", help="Go with soft constraint in leaves") # not used in the paper
parser.add_argument("-d", "--depth", type=int, default=4, help="Final depth of the tree")
parser.add_argument("-lmin", "--min_in_leaves", type=int, default=50, help="Minimal number of points in each leaf")
parser.add_argument("-max", "--max_data", type=int, default=10_000, help="Limit on data inputed into the model")

# optimization parameters
parser.add_argument("-t", "--time_limit", type=int, default=8*3600, help="Total time limit for optimization [s]")
parser.add_argument("-m", "--memory_limit", type=int, default=None, help="Memory limit for gurobi [GB]")
parser.add_argument("-thr", "--n_threads", type=int, default=8, help="Number of threads for gurobi to use")
parser.add_argument("-focus", "--mip_focus", type=int, default=1, help="Value of MIPFocus parameter for Gurobi")
parser.add_argument("-heur", "--mip_heuristics", type=float, default=0.8, help="Value of Heuristics parameter for Gurobi")
args = parser.parse_args()

with open(args.dataset_path, "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {args.dataset_type}")

data_handler = DataHandler(args.dataset_path, round_limit=args.round_limit)
X_train, y_train = data_handler.get_training_data(split_seed=args.random_seed, test_size=0.2, limit=args.max_data)
util = UtilityHelper(data_handler)

logfile_base = args.results_dir + f"/run{args.random_seed}"

print(f"Creating model")
xct = XCT_MIP(args.depth, data_handler, min_in_leaf=args.min_in_leaves, hard_constraint=(not args.soft_constr))
xct.make_model(X_train, y_train)

print("Optimizing the model...")
res = xct.optimize(time_limit=args.time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, mip_focus=args.mip_focus, mip_heuristics=args.mip_heuristics, log_file=f"{logfile_base}.log", verbose=args.verbose)

status = xct.get_humanlike_status()

if res:
    acc = xct.model.getObjective().getValue()

    ctx = xct.get_base_context()
    problem, diff = util.check_leaf_assignment(xct)
    misassigned = np.abs(diff).sum()/2
    ctx["n_misassigned"] = misassigned
    if problem:
        print(f"Problem with solution: {misassigned} points misassigned")
        print("Differences:", diff)

    with open(f"{logfile_base}.ctx", "wb") as f:
        pickle.dump(ctx, f)

    xct.model.write(f"{logfile_base}.sol")
    print(f"Found a solution with {acc*100} leaf accuracy - {status}")
else:
    print(f"Did not find a solution - {status}")
    if status == "INF":
        xct.model.computeIIS()
        xct.model.write(f"{logfile_base}_{status}.ilp")
