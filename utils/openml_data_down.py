import numpy as np
import pandas as pd
import os
import openml
openml.config.cache_directory = os.path.expanduser(os.getcwd() + "/openml_cache")
import pickle

suites_id = {"numerical_classification": 337,
          "categorical_classification": 334}

os.makedirs("data/openml/numerical", exist_ok=True)
os.makedirs("data/openml/categorical", exist_ok=True)

benchmark_suite = openml.study.get_suite(suites_id["numerical_classification"])  # obtain the benchmark suite
for k, task_id in enumerate(benchmark_suite.tasks):  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    print(f"Handling dataset {dataset.name} ({k+1}/{len(benchmark_suite.tasks)})")
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    with open(f"data/openml/numerical/{dataset.name}.pickle", "wb") as f:
        pickle.dump((X, y, categorical_indicator, attribute_names, dataset.name), f)

benchmark_suite = openml.study.get_suite(suites_id["categorical_classification"])  # obtain the benchmark suite
for k, task_id in enumerate(benchmark_suite.tasks):  # iterate over all tasks
    task = openml.tasks.get_task(task_id)  # download the OpenML task
    dataset = task.get_dataset()
    print(f"Handling dataset {dataset.name} ({k+1}/{len(benchmark_suite.tasks)})")
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )

    with open(f"data/openml/categorical/{dataset.name}.pickle", "wb") as f:
        pickle.dump((X, y, categorical_indicator, attribute_names, dataset.name), f)

directory = f"data/openml/categorical/"
for filename in os.listdir(directory):
    with open(directory+filename, "rb") as f:
        X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

    X = np.array(X, dtype=float) # the decision variable must not be a part of data
    y, class_mapping = pd.factorize(y)
    y = np.array(y)

    n_data, n_features = X.shape
    n_classes = len(class_mapping)
    print(f"{dataset_name}, {n_data}, {n_features}, {n_classes}")

directory = f"data/openml/numerical/"
for filename in os.listdir(directory):
    with open(directory+filename, "rb") as f:
        X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)

    X = np.array(X, dtype=float) # the decision variable must not be a part of data
    y, class_mapping = pd.factorize(y)
    y = np.array(y)

    n_data, n_features = X.shape
    n_classes = len(class_mapping)
    print(f"{dataset_name}, {n_data}, {n_features}, {n_classes}")
