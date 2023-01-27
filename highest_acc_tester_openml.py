import numpy as np
import pandas as pd
import time
import os
import pickle
import argparse

from sklearn.model_selection import train_test_split

from XCT import XCT_MIP

parser = argparse.ArgumentParser()
# data parameters
parser.add_argument("-i", "--dataset_i", required=True, type=int, help="Index of the dataset")
parser.add_argument("-type", "--dataset_type", required=True, help="Either categorical or numerical")

# output parameters
parser.add_argument("-res", "--results_dir", required=True, help="Where to store the results")

# model parameters
parser.add_argument("-feas", "--feasibility", action="store_true", help="Aim for feasibility only")
parser.add_argument("-hard", "--hard_constr", action="store_true", help="Go with hard constraint in leaves")
parser.add_argument("-d", "--depth", type=int, default=5, help="Depth of the tree")
parser.add_argument("-max", "--max_data", type=int, default=50_000, help="Limit on data inputed into the model")
# optimization parameters
parser.add_argument("-t", "--time_limit", type=int, default=3600, help="Time limit for optimization [s]")
parser.add_argument("-m", "--memory_limit", type=int, default=None, help="Memory limit for gurobi [GB]")
parser.add_argument("-thr", "--n_threads", type=int, default=None, help="Number of threads for gurobi to use")

# halving method paramters
parser.add_argument("--halving", action="store_true", help="Use the interval halving method")
parser.add_argument("-u", "--upper", type=float, default=1, help="Initial upper bound of the interval halving")
parser.add_argument("-l", "--lower", type=float, default=0.5, help="Initial lower bound of the interval halving")
parser.add_argument("-prec", "--required_prec", type=float, default=0.001, help="Maximal distance between limits upon convergence")

args = parser.parse_args()

directory = f"data/openml/{args.dataset_type}/"
with open(directory+os.listdir(directory)[args.dataset_i], "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {args.dataset_i} in {args.dataset_type}")

X = np.array(X, dtype=float) # the decision variable must not be a part of data
y, class_mapping = pd.factorize(y)
y = np.array(y)

n_classes = len(class_mapping)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train[:args.max_data] # limit the ammount of training data
y_train = y_train[:args.max_data] # limit the ammount of training data

logfile_base = f"{args.results_dir}/{args.dataset_type}/{args.dataset_i}{dataset_name}"
time_limit = args.time_limit
if args.halving:
    print("Starting halving mehtod...")

    high = args.upper
    low = args.lower
    best_model = None
    last_time = time.time()
    while high - low > args.required_prec:
        m = (high+low) / 2
        xct = XCT_MIP(depth=args.depth, leaf_accuracy=m, only_feasibility=args.feasibility,
                    hard_constraint=args.hard_constr)
        xct.prep_model(X_train, n_classes)
        xct.make_model(X_train, y_train)
        res = xct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, log_file=f"{logfile_base}_{m*100:.2f}.log")
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
    xct = XCT_MIP(depth=args.depth)
    xct.prep_model(X_train, n_classes)
    xct.make_model(X_train, y_train)
    print("Optimizing the model...")
    res = xct.optimize(time_limit=time_limit, mem_limit=args.memory_limit, n_threads=args.n_threads, log_file=f"{logfile_base}_direct.log")

    status = xct.get_humanlike_status()

    if res:
        acc = xct.model.getObjective().getValue()

        with open(f"{logfile_base}_{status}_{acc*100:.2f}.ctx", "wb") as f:
            pickle.dump(xct.get_base_context(), f)

        # xct.model.write(f"{logfile_base}_{status}_{acc*100:.2f}.mps")
        xct.model.write(f"{logfile_base}_{status}_{acc*100:.2f}.sol")
        print(f"Found a solution with {acc*100} leaf accuracy - {status}")
    else:
        print(f"Did not find any solution - {status}")
        if status == "INF":
            xct.model.computeIIS()
            xct.model.write(f"{logfile_base}_{status}.ilp")
