import numpy as np
import pandas as pd
import time
import os
import pickle
import argparse

from sklearn.model_selection import train_test_split

from xct_nn.XCT_MIP import XCT_MIP
from xct_nn.XCT_MIP import XCT_MIP

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("-i", "--dataset_i", required=True, type=int, help="Index of the dataset")
parser.add_argument("-type", "--dataset_type", required=True, help="Either categorical or numerical")

# output parameters
parser.add_argument("-res", "--results_dir", required=True, help="Where to store the results")
parser.add_argument("-v", "--verbose",  action="store_true", help="Print model and stats to stdout")

# model parameters
parser.add_argument("-hard", "--hard_constr", action="store_true", help="Go with hard constraint in leaves")
parser.add_argument("-max", "--max_data", type=int, default=1_000, help="Limit on data inputed into the model")
# optimization parameters
parser.add_argument("-t", "--time_limit", type=int, default=18000, help="Time limit for one level [s]")
parser.add_argument("-m", "--memory_limit", type=int, default=None, help="Memory limit for gurobi [GB]")
parser.add_argument("-thr", "--n_threads", type=int, default=None, help="Number of threads for gurobi to use")

args = parser.parse_args()

directory = f"data/openml/{args.dataset_type}/"
with open(directory+os.listdir(directory)[args.dataset_i], "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {args.dataset_i} in {args.dataset_type}")

data_handler = DataHandler(X, y, attribute_names, dataset_name, categorical_indicator)
X_train, y_train = data_handler.get_training_data(split_seed=0, test_size=0.2, limit=args.max_data)

logfile_base = f"{args.results_dir}/{args.dataset_type}/{args.dataset_i}{dataset_name}"
time_limit = args.time_limit
warmstart_values = None
for depth in range(2, 6):
    print(f"Creating model with depth {depth}...")
    xct = XCT_MIP(depth, data_handler, hard_constraint=args.hard_constr)
    xct.make_model(X_train, y_train)
    print("Optimizing the model...")
    res = xct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, log_file=f"{logfile_base}_gradual_{depth}.log", warmstart_values=warmstart_values, verbose=args.verbose)

    status = xct.get_humanlike_status()

    if res:
        acc = xct.model.getObjective().getValue()

        ctx = xct.get_base_context()
        with open(f"{logfile_base}_{depth}_{status}_{acc*100:.2f}.ctx", "wb") as f:
            pickle.dump(ctx, f)

        # prep the values (add depth)
        new_a = np.zeros((X_train.shape[1], 2**(depth+1)-1))
        new_b = np.zeros((2**(depth+1)-1,))
        new_a[:, :(2**depth - 1)] = ctx[0]
        new_a[0, (2**depth - 1):] = 1
        new_b[:(2**depth - 1)] = ctx[1]
        warmstart_values = new_a, new_b

        xct.model.write(f"{logfile_base}_{depth}_{status}_{acc*100:.2f}.sol")
        print(f"Found a solution with {acc*100} leaf accuracy - {status}")
    else:
        print(f"Did not find a solution - {status}")
        if status == "INF":
            xct.model.computeIIS()
            xct.model.write(f"{logfile_base}_{status}.ilp")
        break
