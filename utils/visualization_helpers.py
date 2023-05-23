import pickle
import pandas as pd
import numpy as np
from .datasets import DATASETS_INFO
from scipy.stats import rankdata

DIFFICULTY_THRESH = {"categorical":[0.7, 0.8], "numerical":[0.75, 0.85]}

def get_values(jobs, name, depth=None, starting_tree=False, reduced=False, extended=False, soft=False):
    if "grad" in name and depth is None:
        depth = 4
    suffix = depth if depth is not None else ""
    leaf_suffix = suffix
    if soft:
        leaf_suffix = f"Soft{suffix}"
    prefix = "Start" if starting_tree else ""
    leaf_prefix = prefix + ("Reduced" if reduced or extended else "")
    acc_prefix =  prefix + ("Extended" if extended else ("Reduced" if reduced else ""))
    values = {}
    for dataset_type in DATASETS_INFO:
        values[dataset_type] = {}
        for dataset_name in DATASETS_INFO[dataset_type]:
            datamask = (jobs[name]["DataType"] == dataset_type) & (jobs[name]["DatasetName"] == dataset_name)
            values[dataset_type][dataset_name] = {}
            values[dataset_type][dataset_name]["TrainAcc"] = jobs[name][datamask][f"{acc_prefix}TrainAcc{suffix}"]
            values[dataset_type][dataset_name]["TrainLeafAcc"] = jobs[name][datamask][f"{leaf_prefix}TrainLeafAcc{leaf_suffix}"]
            values[dataset_type][dataset_name]["TestAcc"] = jobs[name][datamask][f"{acc_prefix}TestAcc{suffix}"]
            values[dataset_type][dataset_name]["TestLeafAcc"] = jobs[name][datamask][f"{leaf_prefix}TestLeafAcc{leaf_suffix}"]
            if starting_tree:
                values[dataset_type][dataset_name]["ObjBound"] = jobs[name][datamask][f"{leaf_prefix}TrainLeafAcc{leaf_suffix}"]
            else:
                values[dataset_type][dataset_name]["ObjBound"] = jobs[name][datamask][f"ObjBound{suffix}"]
    return values

def get_values_categ(jobs, name, depth=None, starting_tree=False, reduced=False, extended=False, soft=False):
    vals = get_values(jobs, name, depth, starting_tree, reduced, extended, soft)
    new_vals = {}
    for dtype in vals:
        new_vals[dtype] = {}
        for valname in ["TrainAcc", "TrainLeafAcc", "TestAcc", "TestLeafAcc", "ObjBound"]:
            new_vals[dtype][valname] = pd.concat([d[valname] for d in vals[dtype].values()])
    return new_vals

def gather_by_difficulty(vals):
    new_vals = {}
    for val in vals.values():
        for v in val.values():
            value_names = v.keys()
            break
        break
    for dtype in vals:
        val_lists = {
            "HARD":{valname:[] for valname in value_names},
            "MEDIUM":{valname:[] for valname in value_names},
            "EASY":{valname:[] for valname in value_names},
        }
        for dataset in DATASETS_INFO[dtype]:
            diff = "MEDIUM"
            if DATASETS_INFO[dtype][dataset]["pure_xgb_benchmark_acc"] < DIFFICULTY_THRESH[dtype][0]:
                diff = "HARD"
            elif DATASETS_INFO[dtype][dataset]["pure_xgb_benchmark_acc"] > DIFFICULTY_THRESH[dtype][1]:
                diff = "EASY"
            for valname in value_names:
                val_lists[diff][valname].append(vals[dtype][dataset][valname])
        new_vals[dtype] = {"HARD":{}, "MEDIUM":{}, "EASY":{}}
        for difficulty in val_lists:
            for valname in value_names:
                new_vals[dtype][difficulty][valname] = pd.concat(val_lists[difficulty][valname])
    return new_vals

def get_XGB_dataset_est_values():
    values = {}
    for i, dataset_type in enumerate(DATASETS_INFO):
        values[dataset_type] = {}
        for dataset_name in DATASETS_INFO[dataset_type]:
            val = DATASETS_INFO[dataset_type][dataset_name]["pure_xgb_benchmark_acc"]
            values[dataset_type][dataset_name] = {
                "TrainAcc": pd.Series([], dtype=float),
                "TestAcc": pd.Series([val]),
            }
    return values

def get_XGB_est_values():
    values = {}
    for i, dataset_type in enumerate(DATASETS_INFO):
        values[dataset_type] = {
            "TrainAcc": pd.Series([], dtype=float),
            "TestAcc": pd.Series([], dtype=float),
        }
        for dataset_name in DATASETS_INFO[dataset_type]:
            val = DATASETS_INFO[dataset_type][dataset_name]["pure_xgb_benchmark_acc"]
            values[dataset_type]["TestAcc"] = pd.concat([
                values[dataset_type]["TestAcc"],
                pd.Series([val])
            ])
    return values

def get_tree_results(path, extended=False):
    with open(path, "rb") as f:
        data = pickle.load(f)

    values = {}
    for i, dataset_type in enumerate(DATASETS_INFO):
        values[dataset_type] = {}
        for dataset_name in DATASETS_INFO[dataset_type]:
            if extended:
                values[dataset_type][dataset_name] = {
                    "TrainAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_accs"], dtype=float),
                    "TrainLeafAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_leaf_accs"], dtype=float),
                    "TestAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_test_accs"], dtype=float),
                    "TestLeafAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_test_leaf_accs"], dtype=float),
                    "ObjBound": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_leaf_accs"], dtype=float),
                }
            else:
                values[dataset_type][dataset_name] = {
                    "TrainAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["train_accs"], dtype=float),
                    "TrainLeafAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_leaf_accs"], dtype=float), # this corresponds to leaf accuracy of the reduced tree
                    "TestAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["test_accs"], dtype=float),
                    "TestLeafAcc": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_test_leaf_accs"], dtype=float),
                    "ObjBound": pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_leaf_accs"], dtype=float),
                }
    return values


def get_tree_results_categ(path, extended=False):
    with open(path, "rb") as f:
        data = pickle.load(f)

    values = {}
    for i, dataset_type in enumerate(DATASETS_INFO):
        values[dataset_type] = {
            "TrainAcc": pd.Series([], dtype=float),
            "TrainLeafAcc": pd.Series([], dtype=float),
            "TestAcc": pd.Series([], dtype=float),
            "TestLeafAcc": pd.Series([], dtype=float),
            "ObjBound": pd.Series([], dtype=float),
        }
        for dataset_name in DATASETS_INFO[dataset_type]:
            if extended:
                values[dataset_type]["TrainAcc"] = pd.concat([
                    values[dataset_type]["TrainAcc"],
                    pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_accs"])
                ])
                values[dataset_type]["TestAcc"] = pd.concat([
                    values[dataset_type]["TestAcc"],
                    pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_test_accs"])
                ])
            else:
                values[dataset_type]["TrainAcc"] = pd.concat([
                    values[dataset_type]["TrainAcc"],
                    pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["train_accs"])
                ])
                values[dataset_type]["TestAcc"] = pd.concat([
                    values[dataset_type]["TestAcc"],
                    pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["test_accs"])
                ])
            # ext contains the accuracy after reduction
            values[dataset_type]["TrainLeafAcc"] = pd.concat([
                values[dataset_type]["TrainLeafAcc"],
                pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_leaf_accs"])
            ])
            values[dataset_type]["TestLeafAcc"] = pd.concat([
                values[dataset_type]["TestLeafAcc"],
                pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_test_leaf_accs"])
            ])
            # values[dataset_type]["ObjBound"] = pd.concat([
            #     values[dataset_type]["ObjBound"],
            #     pd.Series(data[f"{dataset_type[0]}_{dataset_name}"]["ext_train_leaf_accs"])
            # ])
    return values

def get_MIP_process(logpath):
    stats = {
        "ObjVal": [],
        "Bound": [],
        "Gap": [],
        "Time": [],
    }
    with open(logpath, "r") as f:
        for l in reversed(f.readlines()):
            if "Nodes" in l:
                break
            if l[0] == "H" or l[0] == "*":
                vals = l.split()
                stats["ObjVal"].insert(0, float(vals[-5]))
                stats["Bound"].insert(0, float(vals[-4]))
                stats["Gap"].insert(0, float(vals[-3][:-1]))
                stats["Time"].insert(0, int(vals[-1][:-1]))

    if not stats["Time"]: # take the initial if no improvement
        # print("empty", logpath)
        with open(logpath, "r") as f:
            for l in reversed(f.readlines()):
                if "Nodes" in l:
                    break
                try:
                    vals = l.split()
                    stats["ObjVal"] = [float(vals[-5])]
                    stats["Bound"] = [float(vals[-4])]
                    stats["Gap"] = [float(vals[-3][:-1])]
                    stats["Time"] = [int(vals[-1][:-1])]
                except:
                    pass
    return stats

def make_ranking(values):
    # values has shape [n_datasets, n_methods]
    # returns mean ranks [n_methods]
    values = np.array(values)
    ranks = np.zeros_like(values)
    for i in range(values.shape[0]):
        ranks[i] = rankdata(-values[i])
    return ranks.mean(axis=0)
