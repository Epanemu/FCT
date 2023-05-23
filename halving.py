# direct approach, but due to poor performance no longer maintained to follow the standards of other runscripts

import time
import os
import pickle
import argparse
import numpy as np

from fct_nn.FCT_MIP import FCT_MIP
from fct_nn.DataHandler import DataHandler
from fct_nn.UtilityHelper import UtilityHelper

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
parser.add_argument("-feas", "--feasibility", action="store_true", help="Aim for feasibility only")
parser.add_argument("-soft", "--soft_constr", action="store_true", help="Go with soft constraint in leaves") # not used in the paper
parser.add_argument("-d", "--depth", type=int, default=4, help="Depth of the tree")
parser.add_argument("-lmin", "--min_in_leaves", type=int, default=50, help="Minimal number of points in each leaf")
parser.add_argument("-max", "--max_data", type=int, default=10_000, help="Limit on data inputed into the model")

# optimization parameters
parser.add_argument("-t", "--time_limit", type=int, default=8*3600, help="Total time limit for optimization [s]")
parser.add_argument("-m", "--memory_limit", type=int, default=None, help="Memory limit for gurobi [GB]")
parser.add_argument("-thr", "--n_threads", type=int, default=8, help="Number of threads for gurobi to use")
parser.add_argument("-focus", "--mip_focus", type=int, default=1, help="Value of MIPFocus parameter for Gurobi")
parser.add_argument("-heur", "--mip_heuristics", type=float, default=0.8, help="Value of Heuristics parameter for Gurobi")

# halving method paramters
parser.add_argument("-u", "--upper", type=float, default=1, help="Initial upper bound of the interval halving")
parser.add_argument("-l", "--lower", type=float, default=0.5, help="Initial lower bound of the interval halving")
parser.add_argument("-prec", "--required_prec", type=float, default=0.001, help="Maximal distance between limits upon convergence")

args = parser.parse_args()

with open(args.dataset_path, "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {args.dataset_type}")

data_handler = DataHandler(args.dataset_path, round_limit=args.round_limit)
X_train, y_train = data_handler.get_training_data(split_seed=args.random_seed, test_size=0.2, limit=args.max_data)
util = UtilityHelper(data_handler)

logfile_base = args.results_dir + f"/run{args.random_seed}"
time_limit = args.time_limit / max(1, np.ceil(np.log2((args.upper-args.lower)/args.required_prec)))
print("Starting halving mehtod...")

high = args.upper
low = args.lower
best_model = None
last_time = time.time()
while high - low > args.required_prec:
    # could be improved by warmstarting with the best so far solution
    m = (high+low) / 2
    fct = FCT_MIP(args.depth, data_handler, min_in_leaf=args.min_in_leaves, leaf_accuracy=m, only_feasibility=args.feasibility, hard_constraint=(not args.soft_constr))
    fct.make_model(X_train, y_train)

    res = fct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, mip_focus=args.mip_focus, mip_heuristics=args.mip_heuristics, log_file=f"{logfile_base}.log", verbose=args.verbose)
    now_time = time.time()

    status = fct.get_humanlike_status()

    if res:
        acc = fct.model.getObjective().getValue()

        ctx = fct.get_base_context()
        problem, diff = util.check_leaf_assignment(fct)
        misassigned = np.abs(diff).sum()/2
        ctx["n_misassigned"] = misassigned
        if problem:
            print(f"Problem with solution: {misassigned} points misassigned")
            print("Differences:", diff)

        with open(f"{logfile_base}_{m*100:.2f}.ctx", "wb") as f:
            pickle.dump(ctx, f)

        best_model = fct.model
        print(f"Found a solution with {acc*100} leaf accuracy - {status}")

    print(f"Attempted {m*100} accuracy - {res} in {(now_time - last_time):.2f} sec")
    last_time = now_time
    if res:
        low = m
    else:
        high = m

if best_model is not None:
    best_model.write(f"{logfile_base}_{low*100:.2f}.sol")

print(f"Accuracy was found between {low*100:.2f}% and {high*100:.2f}%")
print(f"given time limit {time_limit} seconds per model")
print()
