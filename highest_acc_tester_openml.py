# direct approach, but due to poor performance no longer maintained to follow the standards of other runscripts

import time
import os
import pickle
import argparse

from xct_nn.XCT_MIP import XCT_MIP
from xct_nn.DataHandler import DataHandler

parser = argparse.ArgumentParser()
# data parameters
# parser.add_argument("-i", "--dataset_i", required=True, type=int, help="Index of the dataset")
parser.add_argument("-data", "--dataset_path", required=True, help="Path to the dataset")
parser.add_argument("-type", "--dataset_type", required=True, help="Either categorical or numerical")
parser.add_argument("-seed", "--random_seed", type=int, default=0, help="Number with which to seed the data split")
parser.add_argument("-r", "--round_limit", type=int, default=5, help="Max number of decimals in original data")

# output parameters
parser.add_argument("-res", "--results_dir", required=True, help="Where to store the results")
parser.add_argument("-v", "--verbose",  action="store_true", help="Print model and stats to stdout")

# model parameters
parser.add_argument("-feas", "--feasibility", action="store_true", help="Aim for feasibility only")
parser.add_argument("-hard", "--hard_constr", action="store_true", help="Go with hard constraint in leaves")
parser.add_argument("-d", "--depth", type=int, default=5, help="Depth of the tree")
parser.add_argument("-max", "--max_data", type=int, default=50_000, help="Limit on data inputed into the model")
# optimization parameters
parser.add_argument("-t", "--time_limit", type=int, default=3600, help="Time limit for optimization [s]")
parser.add_argument("-m", "--memory_limit", type=int, default=None, help="Memory limit for gurobi [GB]")
parser.add_argument("-thr", "--n_threads", type=int, default=None, help="Number of threads for gurobi to use")
parser.add_argument("-focus", "--mip_focus", type=int, default=0, help="Value of MIPFocus parameter for Gurobi")

# halving method paramters
parser.add_argument("--halving", action="store_true", help="Use the interval halving method")
parser.add_argument("-u", "--upper", type=float, default=1, help="Initial upper bound of the interval halving")
parser.add_argument("-l", "--lower", type=float, default=0.5, help="Initial lower bound of the interval halving")
parser.add_argument("-prec", "--required_prec", type=float, default=0.001, help="Maximal distance between limits upon convergence")

args = parser.parse_args()

# directory = f"data/openml/{args.dataset_type}/"
with open(args.dataset_path, "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {args.dataset_type}")

data_handler = DataHandler(args.dataset_path, round_limit=args.round_limit)
X_train, y_train = data_handler.get_training_data(split_seed=args.random_seed, test_size=0.2, limit=args.max_data)

logfile_base = args.results_dir + f"/run{args.random_seed}"
time_limit = args.time_limit
if args.halving:
    print("Starting halving mehtod...")

    high = args.upper
    low = args.lower
    best_model = None
    last_time = time.time()
    while high - low > args.required_prec:
        m = (high+low) / 2
        xct = XCT_MIP(args.depth, data_handler, leaf_accuracy=m, only_feasibility=args.feasibility,
                    hard_constraint=args.hard_constr)
        xct.make_model(X_train, y_train)
        res = xct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, mip_focus=args.mip_focus, log_file=f"{logfile_base}.log", verbose=args.verbose)
        now_time = time.time()

        if res:
            best_model = xct.model
            with open(f"{logfile_base}_{m*100:.2f}.ctx", "wb") as f:
                pickle.dump(xct.get_base_context(), f)

        print(f"Attempted {m*100} accuracy - {res} in {(now_time - last_time):.2f} sec")
        last_time = now_time
        if res:
            low = m
        else:
            high = m

    if best_model is not None:
        best_model.write(f"{logfile_base}_{low*100:.2f}.mps")
        best_model.write(f"{logfile_base}_{low*100:.2f}.sol")

    print(f"Accuracy was found between {low*100:.2f}% and {high*100:.2f}%")
    print(f"given time limit {time_limit} seconds")
    print()
else:
    print("Creating the model...")
    xct = XCT_MIP(args.depth, data_handler, hard_constraint=args.hard_constr)
    xct.make_model(X_train, y_train)
    print("Optimizing the model...")
    res = xct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, mip_focus=args.mip_focus, log_file=f"{logfile_base}.log", verbose=args.verbose)

    status = xct.get_humanlike_status()

    if res:
        acc = xct.model.getObjective().getValue()

        with open(f"{logfile_base}.ctx", "wb") as f:
            pickle.dump(xct.get_base_context(), f)

        xct.model.write(f"{logfile_base}.sol")
        print(f"Found a solution with {acc*100} leaf accuracy - {status}")
    else:
        print(f"Did not find any solution - {status}")
        if status == "INF":
            xct.model.computeIIS()
            xct.model.write(f"{logfile_base}_{status}.ilp")
            exit(1)
