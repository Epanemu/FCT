import pandas as pd
import pickle

from xct_nn.TreeGenerator import TreeGenerator
from xct_nn.DataHandler import DataHandler

class UtilityHelper:
    def __init__(self, data_handler):
        self.data_h = data_handler
        self.used_X, self.used_y = data_handler.used_data
        self.norm_X = data_handler.normalize(self.used_X)
        self.used_X = data_handler.unnormalize(self.norm_X) # important because of rounding down
        self.tree_gen = TreeGenerator(data_handler)

    def visualize_from_ctx(self, ctx, path, view=False):
        tree = self.tree_gen.make_from_context(ctx)
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y)
        tree.visualize(path, data_handler=self.data_h, view=view)
        tree.reduce_tree(self.data_h)
        tree.visualize_reduced(path+"_red", view, self.data_h)

    def visualize_sklearn(self, skltree, path, hard_constr, view=False):
        # i use own implementation of tree to compute the acc, should be the same to sklearn methods
        tree = self.tree_gen.make_from_sklearn(skltree.tree_, hard_constr, self.norm_X)
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y)
        tree.visualize(path, data_handler=self.data_h, view=view)
        tree.reduce_tree(self.data_h)
        tree.visualize_reduced(path+"_red", view, self.data_h)

    def get_accuracy_from_ctx(self, ctx, X, y):
        tree = self.tree_gen.make_from_context(ctx)
        X = self.data_h.unnormalize(self.data_h.normalize(X))
        return tree.compute_leaf_accuracy(X, y)

    def get_accuracy_sklearn(self, skltree, hard_constr, X, y):
        # i use own implementation of tree to compute the acc, should be the same to sklearn methods
        tree = self.tree_gen.make_from_sklearn(skltree.tree_, hard_constr, self.norm_X)
        X = self.data_h.unnormalize(self.data_h.normalize(X))
        return tree.compute_leaf_accuracy(X, y)

    def check_leaf_assignment(self, xct_mip):
        tree = self.tree_gen.make_from_context(xct_mip.get_base_context())
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y)
        # checking counts is sufficient for my case
        diff = xct_mip.vars["points_in_leaf"].X.round(0) - tree.leaf_totals
        return any(diff != 0), diff

    def n_soft_constrained(self, ctx):
        tree = self.tree_gen.make_from_context(ctx)
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y)
        soft = tree.using_soft_constraint
        if soft is not None:
            return soft.sum()
        return None

    def n_empty_leaves(self, ctx):
        tree = self.tree_gen.make_from_context(ctx)
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y)
        return (tree.leaf_totals == 0).sum()


def get_stats(ctx_path, sklearn_warm=False):
    with open(ctx_path, "rb") as f:
        ctx = pickle.load(f)

    data_h = DataHandler(ctx["data_h_setup"]["path"], ctx["data_h_setup"]["round_limit"])
    data_h.get_training_data(ctx["data_h_setup"]["split_seed"], ctx["data_h_setup"]["test_size"], ctx["data_h_setup"]["limit"], reset_stats=True)
    gen = TreeGenerator(data_h)
    tree = gen.make_from_context(ctx)

    if sklearn_warm:
        sklearn_path = ctx_path[:-4] + "_sklearn.pickle"
        with open(sklearn_path, "rb") as f:
            skltree = pickle.load(f)
        # with soft constraint
        start_tree = gen.make_from_sklearn(skltree.tree_, False, data_h.normalize(data_h.used_data[0]))

    X, y = data_h.used_data
    X = data_h.unnormalize(data_h.normalize(X))
    tr_leaf_acc, tr_acc = tree.compute_leaf_accuracy(X, y)
    if sklearn_warm:
        tr_leaf_acc_s, tr_acc_s = start_tree.compute_leaf_accuracy(X, y)

    X, y = data_h.test_data
    X = data_h.unnormalize(data_h.normalize(X))
    test_leaf_acc, test_acc = tree.compute_leaf_accuracy(X, y)
    if sklearn_warm:
        test_leaf_acc_s, test_acc_s = start_tree.compute_leaf_accuracy(X, y)

    util = UtilityHelper(data_h)
    soft_constr = util.n_soft_constrained(ctx)
    empty_leaves = util.n_empty_leaves(ctx)

    if sklearn_warm:
        return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc, soft_constr, empty_leaves, tr_acc_s, tr_leaf_acc_s, test_acc_s, test_leaf_acc_s
    return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc, soft_constr, empty_leaves

def retrieve_information(base_dir, sklearn_warm=False, gradual_depth=None):
    jobs = pd.read_csv(base_dir+"/jobs", names=["JobName", "JobID", "Path"])
    stats = pd.read_csv(base_dir+"/stats")
    jobs = pd.merge(jobs, stats)

    train_accs = []
    train_leaf_accs = []
    test_accs = []
    test_leaf_accs = []
    soft_constr = []
    empty_leaves = []
    misassigned = []
    gap = []
    obj_bound = []
    status = []
    if sklearn_warm:
        skl_train_accs = []
        skl_train_leaf_accs = []
        skl_test_accs = []
        skl_test_leaf_accs = []
    if gradual_depth is not None:
        for depth in range(1, gradual_depth+1):
            for i, job in jobs.iterrows():
            # for the case when not all depths were computed
            # i = 0
            # while i < len(jobs):
            #     job = jobs.iloc[i]
                ctx_path = job["Path"][:-4] + f"_d{depth}.ctx"
                # import os
                # if not os.path.exists(ctx_path):
                #     print("drop", i)
                #     jobs = jobs.drop([jobs.index[i]])
                #     print(len(jobs))
                #     continue

                with open(ctx_path, "rb") as f:
                    ctx = pickle.load(f)

                if "train_acc" in ctx: # new variant
                    train_accs.append(ctx["train_acc"])
                    train_leaf_accs.append(ctx["train_leaf_acc"])
                    test_accs.append(ctx["test_acc"])
                    test_leaf_accs.append(ctx["test_leaf_acc"])
                    soft_constr.append(ctx["n_soft_constrained"])
                    empty_leaves.append(ctx["n_empty_leaves"])
                    misassigned.append(ctx["n_misassigned"])
                    gap.append(ctx["objective_gap"])
                    obj_bound.append(ctx["objective_bound"])
                    status.append(ctx["status"])
                else:
                    res = get_stats(ctx_path)
                    train_accs.append(res[0])
                    train_leaf_accs.append(res[1])
                    test_accs.append(res[2])
                    test_leaf_accs.append(res[3])
                    soft_constr.append(res[4])
                    empty_leaves.append(res[5])
                # i += 1

            jobs[f"TrainAcc{depth}"] = train_accs
            jobs[f"TrainLeafAcc{depth}"] = train_leaf_accs
            jobs[f"TestAcc{depth}"] = test_accs
            jobs[f"TestLeafAcc{depth}"] = test_leaf_accs
            jobs[f"SoftConstraint{depth}"] = soft_constr
            jobs[f"EmptyLeaves{depth}"] = empty_leaves
            if len(misassigned) > 0:
                jobs[f"MisassignedPoints{depth}"] = misassigned
                jobs[f"ObjGap{depth}"] = gap
                jobs[f"ObjBound{depth}"] = obj_bound
                jobs[f"Status{depth}"] = status
            train_accs = []
            train_leaf_accs = []
            test_accs = []
            test_leaf_accs = []
            soft_constr = []
            empty_leaves = []
            misassigned = []
            gap = []
            obj_bound = []
            status = []
    else:
        for i, job in jobs.iterrows():
            ctx_path = job["Path"][:-4] + ".ctx"

            with open(ctx_path, "rb") as f:
                ctx = pickle.load(f)

            if "train_acc" in ctx:
                train_accs.append(ctx["train_acc"])
                train_leaf_accs.append(ctx["train_leaf_acc"])
                test_accs.append(ctx["test_acc"])
                test_leaf_accs.append(ctx["test_leaf_acc"])
                soft_constr.append(ctx["n_soft_constrained"])
                empty_leaves.append(ctx["n_empty_leaves"])
                misassigned.append(ctx["n_misassigned"])
                gap.append(ctx["objective_gap"])
                obj_bound.append(ctx["objective_bound"])
                status.append(ctx["status"])
                if sklearn_warm:
                    skl_train_accs.append(ctx["train_acc_start"])
                    skl_train_leaf_accs.append(ctx["train_leaf_acc_start"])
                    skl_test_accs.append(ctx["test_acc_start"])
                    skl_test_leaf_accs.append(ctx["test_leaf_acc_start"])
            else:
                res = get_stats(ctx_path, sklearn_warm)
                train_accs.append(res[0])
                train_leaf_accs.append(res[1])
                test_accs.append(res[2])
                test_leaf_accs.append(res[3])
                soft_constr.append(res[4])
                empty_leaves.append(res[5])
                if sklearn_warm:
                    skl_train_accs.append(res[6])
                    skl_train_leaf_accs.append(res[7])
                    skl_test_accs.append(res[8])
                    skl_test_leaf_accs.append(res[9])

        jobs["TrainAcc"] = train_accs
        jobs["TrainLeafAcc"] = train_leaf_accs
        jobs["TestAcc"] = test_accs
        jobs["TestLeafAcc"] = test_leaf_accs
        jobs["SoftConstraint"] = soft_constr
        jobs["EmptyLeaves"] = empty_leaves
        if len(misassigned) > 0:
            jobs["MisassignedPoints"] = misassigned
            jobs["ObjGap"] = gap
            jobs["ObjBound"] = obj_bound
            jobs["Status"] = status
        if sklearn_warm:
            jobs["StartTrainAcc"] = skl_train_accs
            jobs["StartTrainLeafAcc"] = skl_train_leaf_accs
            jobs["StartTestAcc"] = skl_test_accs
            jobs["StartTestLeafAcc"] = skl_test_leaf_accs

    data_types = []
    dataset_names = []
    seeds = []
    memory = []
    for i, job in jobs.iterrows():
        path_elements = job["Path"].split("/")
        data_types.append(path_elements[-3])
        dataset_names.append(path_elements[-2])
        seeds.append(path_elements[-1].split(".")[0][3:]) # seed number
        memory.append(float(job["MemRSS"][:-1]))
        if job["MemRSS"][-1] == "K":
            memory[-1] /= 1024 * 1024
        if job["MemRSS"][-1] == "M":
            memory[-1] /= 1024

    jobs["DataType"] = data_types
    jobs["DatasetName"] = dataset_names
    jobs["Seed"] = seeds
    jobs["MemRSS"] = memory
    return jobs
