import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import os
import sys
import pickle
# import openml
# openml.config.cache_directory = os.path.expanduser(os.getcwd() + "/openml_cache")

from DecisionTree import DecisionTreeMIP


# suites_id = {"numerical_regression": 297,
#           "numerical_classification": 298,
#           "categorical_regression": 299,
#           "categorical_classification": 304}

# benchmark_suite = openml.study.get_suite(suites_id[sys.argv[1]])  # obtain the benchmark suite
# # for k, task_id in enumerate(benchmark_suite.tasks):  # iterate over all tasks

# k = int(sys.argv[2])
# task_id = benchmark_suite.tasks[k]

# task = openml.tasks.get_task(task_id)  # download the OpenML task
# dataset = task.get_dataset()
# print(f"Handling dataset {dataset.name} ({k+1}/{len(benchmark_suite.tasks)})")
# X, y, categorical_indicator, attribute_names = dataset.get_data(
#     dataset_format="dataframe", target=dataset.default_target_attribute
# )

directory = f"data/openml/{sys.argv[1]}/"
with open(directory+os.listdir(directory)[int(sys.argv[2])], "rb") as f:
    X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

print(f"Handling dataset {dataset_name} - {sys.argv[2]} in {sys.argv[1]}")

X = np.array(X, dtype=float) # the decision variable must not be a part of data
y, class_mapping = pd.factorize(y)
y = np.array(y)

n_data, n_features = X.shape
n_classes = len(class_mapping)

scales = np.empty((n_features,))
epsilons = np.empty((n_features,))
for i, col_data in enumerate(X.T):
    scales[i] = col_data.max() # assumes non-negative numbers -TODO improve that?
    # would more effective to not need to compute this for those arbitrarily set
    # the time spent on this is negligible in comparison to the MIP optimization though...
    col_sorted = col_data.copy()
    col_sorted.sort()
    eps = col_sorted[1:] - col_sorted[:-1]
    eps[eps == 0] = np.inf
    epsilons[i] = eps.min()

scales[scales == 0] = 1 # to not divide by zero, if all values were 0
epsilons[epsilons == np.inf] = 0 # if all values were same, we actually want eps 0

epsilons /= scales
X /= scales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_test = X_test[:50_000] # hard limit on the ammount of data
y_test = y_test[:50_000] # hard limit on the ammount of data

print("Starting testing...")

time_limit = 3600
high = 1
low = 0.5
last_time = time.time()
best_model = None
while high - low > 0.001:
    m = (high+low) / 2
    dt = DecisionTreeMIP(depth=5, leaf_accuracy=m)
    res, model = dt.fit_model(X_train, y_train, n_classes, epsilons, time_limit=time_limit,
        log_file=f"openml_solutions_hard_feas/{sys.argv[1]}/{sys.argv[2]}{dataset_name}_{m*100:.2f}.log")
    now_time = time.time()

    if res:
        best_model = model

    print(f"Attempted {m*100} accuracy - {res} in {(now_time - last_time):.2f} sec")
    last_time = now_time
    if res:
        low = m
    else:
        high = m

if best_model is not None:
    best_model.write(f"openml_solutions_hard_feas/{sys.argv[1]}/{sys.argv[2]}{dataset_name}_{low*100:.2f}.mps")
    best_model.write(f"openml_solutions_hard_feas/{sys.argv[1]}/{sys.argv[2]}{dataset_name}_{low*100:.2f}.sol")

print(f"Accuracy was found between {low*100:.2f}% and {high*100:.2f}%")
print(f"given time limit {time_limit} seconds")
print()