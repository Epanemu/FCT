import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

from DecisionTree import DecisionTreeMIP


data_path = "data/german_credit_data.csv"
data = pd.read_csv(data_path)
# MAP STRING VALUES TO NUMBERS
data.fillna("not applicable", inplace=True)

val_map = {}
# The ordering matters for some values...
# data["Saving accounts"], val_map["Saving accounts"] = pd.factorize(data["Saving accounts"])
# data["Checking account"], val_map["Checking account"] = pd.factorize(data["Checking account"])
val_map["Saving accounts"] = ['not applicable', 'little', 'moderate', 'quite rich', 'rich']
data["Saving accounts"] =  data["Saving accounts"].map({val:i for i, val in enumerate(val_map["Saving accounts"])})
val_map["Checking account"] = ['not applicable', 'little', 'moderate', 'rich']
data["Checking account"] =  data["Checking account"].map({val:i for i, val in enumerate(val_map["Checking account"])})

# binary variables
data["Sex"], val_map["Sex"] = pd.factorize(data["Sex"])
data["Risk"], val_map["Risk"] = pd.factorize(data["Risk"])

# TODO: make strictly categorical vars to one hot - would be likely better
data["Housing"], val_map["Housing"] = pd.factorize(data["Housing"])
data["Purpose"], val_map["Purpose"] = pd.factorize(data["Purpose"])

# prepare values to [0,1] range, find epsilons

# drop index
data.drop(data.columns[[0]], axis=1, inplace=True)

X = np.array(data[data.columns[:-1]], dtype=float) # the decision variable must not be a part of data
y = np.array(data["Risk"])

n_data, n_features = X.shape
n_classes = len(val_map["Risk"])

scales = np.empty((n_features,))
# TODO figure out epsilons inside DTMIP
epsilons = np.empty((n_features,))
for i, col_data in enumerate(X.T):
    scales[i] = col_data.max()
    # would more effective to not need to compute this for those arbitrarily set
    # the time spent on this is negligible in comparison to the MIP optimization though...
    col_sorted = col_data.copy()
    col_sorted.sort()
    eps = col_sorted[1:] - col_sorted[:-1]
    eps[eps == 0] = np.inf
    epsilons[i] = eps.min()

scales[scales == 0] = 1
epsilons[epsilons == np.inf] = 0

epsilons /= scales
# epsilons[...] = 0.005 # ADDED WORKAROUND FOR FEASIBILITY OF WARMSTART
# problem je v tom, ze generovanej warmstart nahodi bcka takovy, ze se pak bod nevejde pod ne s pomoci prilis velkyho epsilonu, ackoli na tu stranu patri
# epsilons /= 2 # TODO ADDED!
# TODO what is the gurobi output
# TODO warm start using sklearn
X /= scales

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

up = 0.8
low = 0.5
last_time = time.time()
while up - low > 0.001:
    m = (up+low) / 2
    dt = DecisionTreeMIP(depth=5, leaf_accuracy=m)
    res = dt.fit_model(X_train, y_train, n_classes, epsilons, time_limit=10)
    now_time = time.time()
    print(f"Attempted {m*100} accuracy - {res} in {(now_time - last_time):.2f} sec")
    last_time = now_time
    if res:
        low = m
    else:
        up = m