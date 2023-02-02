import os
import pickle
import numpy as np

from xct_nn.TreeGenerator import TreeGenerator

# folders = ["results/from_rci/direct_categorical/", "results/from_rci/feas_categorical/", "results/from_rci/minim_categorical/", "results/from_rci/gradual_categorical/"]
folders = ["results/from_rci/gradual_categorical/"]


def get_context(context_path):
    with open(context_path, "rb") as f:
        ctx = pickle.load(f)
    if len(ctx) == 5:
        return ctx

    elif len(ctx) == 1:
        return ctx

    elif len(ctx) == 7:
        a, b, classes, shifts, scales, epsilons, depth = ctx

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from xct_nn.DataHandler import DataHandler

        dataset_name = "_".join(context_path.split("/")[-1].split("_")[:-2])
        if dataset_name[1] in "0123456789":
            dataset_name = dataset_name[2:]
        else:
            dataset_name = dataset_name[1:]
        if dataset_name[-2] == "_" and dataset_name[-1] in "12345":
            dataset_name = dataset_name[:-2]
        data_path = f"data/openml/categorical/{dataset_name}.pickle"

        if "direct" in context_path or "gradual" in context_path:
            data_limit = 1_000
        elif "10k" in context_path:
            data_limit = 10_000
        else:
            data_limit = 50_000
        data_limit = 10_000

        with open(data_path, "rb") as f:
            X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)
        data_h = DataHandler(X, y, attribute_names, dataset_name, categorical_indicator)
        data_h.get_training_data(split_seed=0, test_size=0.2, limit=data_limit)

        data_h.set_normalizers(shifts, scales)
        # dunno why this does not work for most...
        # assert np.all(shifts == data_h.shifts)
        # assert np.all(scales == data_h.scales)
        # assert np.all(epsilons == data_h.epsilons)

        # X = np.array(X, dtype=float) # the decision variable must not be a part of data
        # y, class_mapping = pd.factorize(y)
        # y = np.array(y)
        # n_classes = len(class_mapping)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        return a, b, classes, data_h, depth


    else:
        # SOME OLD VARIANT, probably not used anywhere
        scales, shifts, a, b = ctx

        import pandas as pd
        from sklearn.model_selection import train_test_split
        from xct_nn.DataHandler import DataHandler

        dataset_name = "_".join(context_path.split("/")[-1].split("_")[:-1])
        if dataset_name[1] in "0123456789":
            dataset_name = dataset_name[2:]
        else:
            dataset_name = dataset_name[1:]
        data_path = f"data/openml/categorical/{dataset_name}.pickle"

        if "10k" in context_path:
            data_limit = 10_000
        else:
            data_limit = 50_000
        data_limit = 10_000

        with open(data_path, "rb") as f:
            X, y, categorical_indicator, attribute_names, dataset_name = pickle.load(f)
        data_h = DataHandler(X, y, attribute_names, dataset_name, categorical_indicator)
        data_h.get_training_data(split_seed=0, test_size=0.2, limit=data_limit)

        data_h.set_normalizers(shifts, scales)

        return a, b, None, data_h, None


for folder in folders:
    # print(folder)
    for filename in os.listdir(folder):
        if not filename.endswith(".ctx"):
            continue
        # print(filename)
        ctx = get_context(folder+filename)

        if len(ctx) == 3:
            _, _, data_h = ctx
            gen = TreeGenerator(data_h)
            path = folder+filename[:-4]+".sol"
            tree = gen.make_from_SOL_file(path)
        elif len(ctx) == 5:
            a, b, classes, data_h, depth = ctx
            gen = TreeGenerator(data_h)
            tree = gen.make_from_matrices(a, b, classes, depth)
        else:
            gen = TreeGenerator(ctx["data_h"])
            tree = gen.make_from_context(ctx)
        # print(data_h.dataset_name)
        # print("Train:", tree.compute_accuracy2(*data_h.train_data))
        # print("Test:", tree.compute_accuracy2(*data_h.test_data))
        # print()
        leaf_train, tot_train = tree.compute_leaf_accuracy(*data_h.train_data)
        leaf_test, tot_test = tree.compute_leaf_accuracy(*data_h.test_data)
        print(f"{folder},{filename},{leaf_train*100},{tot_train*100},{leaf_test*100},{tot_test*100}")