import time
import os
import pickle
import argparse
import numpy as np
from sklearn import tree as skltree

from xct_nn.XCT_MIP import XCT_MIP
from xct_nn.DataHandler import DataHandler
from xct_nn.TreeGenerator import TreeGenerator
from xct_nn.UtilityHelper import UtilityHelper

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("-data", "--dataset_path", required=True, help="Path to the dataset")
parser.add_argument("-type", "--dataset_type", required=True, help="Either categorical or numerical")
parser.add_argument("-seed", "--random_seed", type=int, default=0, help="Number with which to seed the data split and sklearn model")
parser.add_argument("-r", "--round_limit", type=int, default=5, help="Max number of decimals in original data")

# output parameters
parser.add_argument("-res", "--results_dir", required=True, help="Where to store the results")
parser.add_argument("-v", "--verbose",  action="store_true", help="Print model and stats to stdout")

# model parameters
parser.add_argument("-hard", "--hard_constr", action="store_true", help="Go with hard constraint in leaves")
parser.add_argument("-init", "--init_type", default="warmstart", help="How should the values of sklearn tree be used? [warmstart, hint, fix_values]")
parser.add_argument("-d", "--depth", type=int, default=5, help="Depth of the tree")
parser.add_argument("-max", "--max_data", type=int, default=50_000, help="Limit on data inputed into the model")

# optimization parameters
parser.add_argument("-t", "--time_limit", type=int, default=3600, help="Time limit for optimization [s]")
parser.add_argument("-m", "--memory_limit", type=int, default=None, help="Memory limit for gurobi [GB]")
parser.add_argument("-thr", "--n_threads", type=int, default=None, help="Number of threads for gurobi to use")
parser.add_argument("-focus", "--mip_focus", type=int, default=0, help="Value of MIPFocus parameter for Gurobi")
parser.add_argument("-heur", "--mip_heuristics", type=float, default=0.05, help="Value of Heuristics parameter for Gurobi")

args = parser.parse_args()

with open(args.dataset_path, "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {args.dataset_type}")

data_handler = DataHandler(args.dataset_path, round_limit=args.round_limit)
X_train, y_train = data_handler.get_training_data(split_seed=args.random_seed, test_size=0.2, limit=args.max_data)
util = UtilityHelper(data_handler)

logfile_base = args.results_dir + f"/run{args.random_seed}"
time_limit = args.time_limit

print("Preparing sklearn model...")
dt_sklearn = skltree.DecisionTreeClassifier(max_depth=args.depth, random_state=args.random_seed)
dt_sklearn.fit(X_train, y_train)
with open(f"{logfile_base}_sklearn.pickle", "wb") as f:
    pickle.dump(dt_sklearn, f)

tree_gen = TreeGenerator(data_handler)
tree = tree_gen.make_from_sklearn(dt_sklearn.tree_, soft_limit=20, train_data=X_train)
values = tree.as_ab_values()

print("Creating the MIP model...")
xct = XCT_MIP(args.depth, data_handler, hard_constraint=args.hard_constr)
xct.make_model(X_train, y_train)

print("Optimizing the model...")
res = xct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, mip_focus=args.mip_focus, mip_heuristics=args.mip_heuristics, log_file=f"{logfile_base}.log", initialize=args.init_type, values=values, verbose=args.verbose)

status = xct.get_humanlike_status()

if res:
    acc = xct.model.getObjective().getValue()
    print(f"Found a solution with {acc*100} leaf accuracy - {status}")

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
else:
    print(f"Did not find any solution - {status}")
    if status == "INF":
        xct.model.computeIIS()
        xct.model.write(f"{logfile_base}_{status}.ilp")
        exit(1)
