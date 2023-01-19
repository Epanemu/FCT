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
parser.add_argument("-t", "--time_limit", type=int, default=3600, help="Time limit for optimization")

# halving method paramters
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

n_data, n_features = X.shape
n_classes = len(class_mapping)

shifts = np.empty((n_features,))
scales = np.empty((n_features,))
epsilons = np.empty((n_features,))
for i, col_data in enumerate(X.T):
    shifts[i] = col_data.min()
    col_data -= shifts[i]
    scales[i] = col_data.max()
    if scales[i] == 0:
        scales[i] = 1 # to not divide by zero, if all values were the same
    col_data /= scales[i]
    # would more effective to not need to compute this for those arbitrarily set
    # the time spent on this is negligible in comparison to the MIP optimization though...
    col_sorted = col_data.copy()
    col_sorted.sort()
    eps = col_sorted[1:] - col_sorted[:-1]
    eps[eps == 0] = np.inf
    epsilons[i] = eps.min()

epsilons[epsilons == np.inf] = 1 # if all values were same, we actually want eps nonzero to prevent false splitting

assert np.all(epsilons > 0)
assert np.all((X >= 0) & (X <= 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train = X_train[:args.max_data] # hard limit on the ammount of data
y_train = y_train[:args.max_data] # hard limit on the ammount of data

print("Starting testing...")

time_limit = args.time_limit
high = args.upper
low = args.lower
last_time = time.time()
best_model = None
while high - low > args.required_prec:
    m = (high+low) / 2
    xct = XCT_MIP(depth=args.depth, leaf_accuracy=m, only_feasibility=args.feasibility,
                 hard_constraint=args.hard_constr)
    res, model, a, b = xct.fit_model(X_train, y_train, n_classes, epsilons, time_limit=time_limit,
        log_file=f"{args.results_dir}/{args.dataset_type}/{args.dataset_i}{dataset_name}_{m*100:.2f}.log")
    now_time = time.time()

    if res:
        best_model = model
        with open(f"{args.results_dir}/{args.dataset_type}/{args.dataset_i}{dataset_name}_{low*100:.2f}.ctx", "wb") as f:
            pickle.dump((scales, shifts, a.X, b.X), f)

    print(f"Attempted {m*100} accuracy - {res} in {(now_time - last_time):.2f} sec")
    last_time = now_time
    if res:
        low = m
    else:
        high = m

if best_model is not None:
    best_model.write(f"{args.results_dir}/{args.dataset_type}/{args.dataset_i}{dataset_name}_{low*100:.2f}.mps")
    best_model.write(f"{args.results_dir}/{args.dataset_type}/{args.dataset_i}{dataset_name}_{low*100:.2f}.sol")

print(f"Accuracy was found between {low*100:.2f}% and {high*100:.2f}%")
print(f"given time limit {time_limit} seconds")
print()