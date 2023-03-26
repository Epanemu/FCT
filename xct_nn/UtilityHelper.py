import pandas as pd
import pickle

from xct_nn.TreeGenerator import TreeGenerator
from xct_nn.DataHandler import DataHandler
from xct_nn.XCT_Extended import XCT_Extended

class UtilityHelper:
    def __init__(self, data_handler):
        self.data_h = data_handler
        self.used_X, self.used_y = data_handler.used_data
        self.norm_X = data_handler.normalize(self.used_X)
        self.used_X = data_handler.unnormalize(self.norm_X) # important because of rounding down
        self.tree_gen = TreeGenerator(data_handler)

    def visualize_from_ctx(self, ctx, path, view=False, visualize_test=False):
        tree = self.tree_gen.make_from_context(ctx)
        soft_limit = ctx["leaf_acc_limit"] if not ctx["hard_constraint"] else 0
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y, soft_limit=soft_limit)
        tree.visualize(path, data_handler=self.data_h, view=view)
        tree.reduce_tree(self.data_h)
        tree.visualize_reduced(path+"_red", view, self.data_h)
        if visualize_test:
            X, y = self.data_h.test_data
            X = self.data_h.unnormalize(self.data_h.normalize(X))
            _ = tree.compute_leaf_accuracy_reduced(X, y, soft_limit=soft_limit)
            tree.visualize_reduced(path+"_red_test", view, self.data_h)

    def visualize_sklearn(self, skltree, path, soft_limit=0, view=False):
        # i use own implementation of tree to compute the acc, should be the same to sklearn methods
        tree = self.tree_gen.make_from_sklearn(skltree.tree_, soft_limit, self.norm_X)
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y, soft_limit=soft_limit)
        tree.visualize(path, data_handler=self.data_h, view=view)
        tree.reduce_tree(self.data_h)
        tree.visualize_reduced(path+"_red", view, self.data_h)

    def get_accuracy_from_ctx(self, ctx, X, y, soft_limit=0):
        tree = self.tree_gen.make_from_context(ctx)
        X = self.data_h.unnormalize(self.data_h.normalize(X))
        return tree.compute_leaf_accuracy(X, y, soft_limit=soft_limit)

    def get_accuracy_sklearn(self, skltree, soft_limit, X, y):
        # i use own implementation of tree to compute the acc, should be the same to sklearn methods
        tree = self.tree_gen.make_from_sklearn(skltree.tree_, soft_limit, self.norm_X)
        X = self.data_h.unnormalize(self.data_h.normalize(X))
        return tree.compute_leaf_accuracy(X, y, soft_limit=soft_limit)

    def check_leaf_assignment(self, xct_mip):
        tree = self.tree_gen.make_from_context(xct_mip.get_base_context())
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y)
        # checking counts is sufficient for my case
        diff = xct_mip.vars["points_in_leaf"].X.round(0) - tree.leaf_totals
        return any(diff != 0), diff

    def n_soft_constrained(self, ctx):
        if not ctx["hard_constraint"]:
            tree = self.tree_gen.make_from_context(ctx)
            _ = tree.compute_leaf_accuracy(self.used_X, self.used_y, soft_limit=ctx["leaf_acc_limit"])
            soft = tree.using_soft_constraint
            if soft is not None:
                return soft.sum()
        return None

    def n_empty_leaves(self, ctx):
        tree = self.tree_gen.make_from_context(ctx)
        _ = tree.compute_leaf_accuracy(self.used_X, self.used_y)
        return (tree.leaf_totals == 0).sum()


def get_stats(ctx_path, data_h, sklearn_warm=False, soft_limit=0):
    with open(ctx_path, "rb") as f:
        ctx = pickle.load(f)

    gen = TreeGenerator(data_h)
    tree = gen.make_from_context(ctx)

    if sklearn_warm:
        sklearn_path = ctx_path[:-4] + "_sklearn.pickle"
        with open(sklearn_path, "rb") as f:
            skltree = pickle.load(f)
        # with soft constraint
        start_tree = gen.make_from_sklearn(skltree.tree_, soft_limit, data_h.normalize(data_h.used_data[0]))

    X, y = data_h.used_data
    X = data_h.unnormalize(data_h.normalize(X))
    tr_leaf_acc, tr_acc = tree.compute_leaf_accuracy(X, y, soft_limit=soft_limit)
    if sklearn_warm:
        tr_leaf_acc_s, tr_acc_s = start_tree.compute_leaf_accuracy(X, y, soft_limit=soft_limit)

    X, y = data_h.test_data
    X = data_h.unnormalize(data_h.normalize(X))
    test_leaf_acc, test_acc = tree.compute_leaf_accuracy(X, y, soft_limit=soft_limit)
    if sklearn_warm:
        test_leaf_acc_s, test_acc_s = start_tree.compute_leaf_accuracy(X, y, soft_limit=soft_limit)

    if sklearn_warm:
        return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc, tr_acc_s, tr_leaf_acc_s, test_acc_s, test_leaf_acc_s
    return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc


def get_reduced_stats(ctx_path, data_h, sklearn_warm=False, soft_limit=0):
    with open(ctx_path, "rb") as f:
        ctx = pickle.load(f)

    gen = TreeGenerator(data_h)
    tree = gen.make_from_context(ctx)
    tree.reduce_tree(data_h)

    if sklearn_warm:
        sklearn_path = ctx_path[:-4] + "_sklearn.pickle"
        with open(sklearn_path, "rb") as f:
            skltree = pickle.load(f)
        # with soft constraint
        start_tree = gen.make_from_sklearn(skltree.tree_, soft_limit, data_h.normalize(data_h.used_data[0]))
        start_tree.reduce_tree(data_h)

    X, y = data_h.used_data
    X = data_h.unnormalize(data_h.normalize(X))
    tr_leaf_acc, tr_acc = tree.compute_leaf_accuracy_reduced(X, y, soft_limit=soft_limit)
    if sklearn_warm:
        tr_leaf_acc_s, tr_acc_s = start_tree.compute_leaf_accuracy_reduced(X, y, soft_limit=soft_limit)

    X, y = data_h.test_data
    X = data_h.unnormalize(data_h.normalize(X))
    test_leaf_acc, test_acc = tree.compute_leaf_accuracy_reduced(X, y, soft_limit=soft_limit)
    if sklearn_warm:
        test_leaf_acc_s, test_acc_s = start_tree.compute_leaf_accuracy_reduced(X, y, soft_limit=soft_limit)

    if sklearn_warm:
        return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc, tr_acc_s, tr_leaf_acc_s, test_acc_s, test_leaf_acc_s
    return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc


def get_extended_stats(ctx_path, data_h, seed, sklearn_warm=False, soft_limit=0):
    # TODO make this load the extended model from a file...
    with open(ctx_path, "rb") as f:
        ctx = pickle.load(f)

    gen = TreeGenerator(data_h)
    tree = gen.make_from_context(ctx)
    ext_tree = XCT_Extended(tree, data_h, seed=seed, search_iterations=1)

    if sklearn_warm:
        sklearn_path = ctx_path[:-4] + "_sklearn.pickle"
        with open(sklearn_path, "rb") as f:
            skltree = pickle.load(f)
        # with soft constraint
        start_tree = gen.make_from_sklearn(skltree.tree_, soft_limit, data_h.normalize(data_h.used_data[0]))
        ext_start_tree = XCT_Extended(start_tree, data_h, seed=seed, search_iterations=1)

    X, y = data_h.used_data
    X = data_h.unnormalize(data_h.normalize(X))
    tr_leaf_acc, tr_acc = ext_tree.compute_accuracy(X, y, soft_limit=soft_limit)
    if sklearn_warm:
        tr_leaf_acc_s, tr_acc_s = ext_start_tree.compute_accuracy(X, y, soft_limit=soft_limit)

    X, y = data_h.test_data
    X = data_h.unnormalize(data_h.normalize(X))
    test_leaf_acc, test_acc = ext_tree.compute_accuracy(X, y, soft_limit=soft_limit)
    if sklearn_warm:
        test_leaf_acc_s, test_acc_s = ext_start_tree.compute_accuracy(X, y, soft_limit=soft_limit)

    if sklearn_warm:
        return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc, tr_acc_s, tr_leaf_acc_s, test_acc_s, test_leaf_acc_s
    return tr_acc, tr_leaf_acc, test_acc, test_leaf_acc


def retrieve_information(base_dir, sklearn_warm=False, gradual_depth=None, soft_bound=20):
    jobs = pd.read_csv(base_dir+"/jobs", names=["JobName", "JobID", "Path"])
    stats = pd.read_csv(base_dir+"/stats")
    jobs = pd.merge(jobs, stats)

    train_accs = []
    train_leaf_accs = []
    train_leaf_soft_accs = []
    test_accs = []
    test_leaf_accs = []
    test_leaf_soft_accs = []
    red_train_accs = []
    red_train_leaf_accs = []
    red_train_leaf_soft_accs = []
    red_test_accs = []
    red_test_leaf_accs = []
    red_test_leaf_soft_accs = []
    extend_train_accs = []
    extend_test_accs = []
    soft_constr = []
    empty_leaves = []
    misassigned = []
    gap = []
    obj_bound = []
    status = []
    if sklearn_warm:
        skl_train_accs = []
        skl_train_leaf_accs = []
        skl_train_leaf_soft_accs = []
        skl_test_accs = []
        skl_test_leaf_accs = []
        skl_test_leaf_soft_accs = []
        skl_red_train_accs = []
        skl_red_train_leaf_accs = []
        skl_red_train_leaf_soft_accs = []
        skl_red_test_accs = []
        skl_red_test_leaf_accs = []
        skl_red_test_leaf_soft_accs = []
        skl_extend_train_accs = []
        skl_extend_test_accs = []

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

                dh_setup = ctx["data_h_setup"]
                data_h = DataHandler(dh_setup["path"], dh_setup["round_limit"])
                data_h.get_training_data(dh_setup["split_seed"], dh_setup["test_size"], dh_setup["limit"], reset_stats=True)

                util = UtilityHelper(data_h)
                soft_constr.append(util.n_soft_constrained(ctx))
                empty_leaves.append(util.n_empty_leaves(ctx))

                misassigned.append(ctx["n_misassigned"])
                gap.append(ctx["objective_gap"])
                obj_bound.append(ctx["objective_bound"])
                status.append(ctx["status"])

                res = get_stats(ctx_path, data_h, soft_limit=0)
                train_accs.append(res[0])
                train_leaf_accs.append(res[1])
                test_accs.append(res[2])
                test_leaf_accs.append(res[3])

                s_res = get_stats(ctx_path, data_h, soft_limit=soft_bound)
                train_leaf_soft_accs.append(s_res[1])
                test_leaf_soft_accs.append(s_res[3])

                red_res = get_reduced_stats(ctx_path, data_h, soft_limit=0)
                red_train_accs.append(red_res[0])
                red_train_leaf_accs.append(red_res[1])
                red_test_accs.append(red_res[2])
                red_test_leaf_accs.append(red_res[3])

                red_s_res = get_reduced_stats(ctx_path, data_h, soft_limit=soft_bound)
                red_train_leaf_soft_accs.append(red_s_res[1])
                red_test_leaf_soft_accs.append(red_s_res[3])

                train, _, test, _ = get_extended_stats(ctx_path, data_h, seed=ctx["data_h_setup"]["split_seed"])
                extend_train_accs.append(train)
                extend_test_accs.append(test)

            jobs[f"TrainAcc{depth}"] = train_accs
            jobs[f"TrainLeafAcc{depth}"] = train_leaf_accs
            jobs[f"TrainLeafAccSoft{depth}"] = train_leaf_soft_accs
            jobs[f"TestAcc{depth}"] = test_accs
            jobs[f"TestLeafAcc{depth}"] = test_leaf_accs
            jobs[f"TestLeafAccSoft{depth}"] = test_leaf_soft_accs
            jobs[f"ReducedTrainAcc{depth}"] = red_train_accs
            jobs[f"ReducedTrainLeafAcc{depth}"] = red_train_leaf_accs
            jobs[f"ReducedTrainLeafAccSoft{depth}"] = red_train_leaf_soft_accs
            jobs[f"ReducedTestAcc{depth}"] = red_test_accs
            jobs[f"ReducedTestLeafAcc{depth}"] = red_test_leaf_accs
            jobs[f"ReducedTestLeafAccSoft{depth}"] = red_test_leaf_soft_accs
            jobs[f"ExtendedTrainAcc{depth}"] = extend_train_accs
            jobs[f"ExtendedTestAcc{depth}"] = extend_test_accs
            jobs[f"SoftConstraint{depth}"] = soft_constr
            jobs[f"EmptyLeaves{depth}"] = empty_leaves
            jobs[f"MisassignedPoints{depth}"] = misassigned
            jobs[f"ObjGap{depth}"] = gap
            jobs[f"ObjBound{depth}"] = obj_bound
            jobs[f"Status{depth}"] = status

            train_accs = []
            train_leaf_accs = []
            train_leaf_soft_accs = []
            test_accs = []
            test_leaf_accs = []
            test_leaf_soft_accs = []
            red_train_accs = []
            red_train_leaf_accs = []
            red_train_leaf_soft_accs = []
            red_test_accs = []
            red_test_leaf_accs = []
            red_test_leaf_soft_accs = []
            extend_train_accs = []
            extend_test_accs = []
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

            dh_setup = ctx["data_h_setup"]
            data_h = DataHandler(dh_setup["path"], dh_setup["round_limit"])
            data_h.get_training_data(dh_setup["split_seed"], dh_setup["test_size"], dh_setup["limit"], reset_stats=True)

            util = UtilityHelper(data_h)
            soft_constr.append(util.n_soft_constrained(ctx))
            empty_leaves.append(util.n_empty_leaves(ctx))

            misassigned.append(ctx["n_misassigned"])
            gap.append(ctx["objective_gap"])
            obj_bound.append(ctx["objective_bound"])
            status.append(ctx["status"])

            res = get_stats(ctx_path, data_h, sklearn_warm, soft_limit=0)
            train_accs.append(res[0])
            train_leaf_accs.append(res[1])
            test_accs.append(res[2])
            test_leaf_accs.append(res[3])

            s_res = get_stats(ctx_path, data_h, sklearn_warm, soft_limit=soft_bound)
            train_leaf_soft_accs.append(s_res[1])
            test_leaf_soft_accs.append(s_res[3])

            red_res = get_reduced_stats(ctx_path, data_h, sklearn_warm, soft_limit=0)
            red_train_accs.append(red_res[0])
            red_train_leaf_accs.append(red_res[1])
            red_test_accs.append(red_res[2])
            red_test_leaf_accs.append(red_res[3])

            red_s_res = get_reduced_stats(ctx_path, data_h, sklearn_warm, soft_limit=soft_bound)
            red_train_leaf_soft_accs.append(red_s_res[1])
            red_test_leaf_soft_accs.append(red_s_res[3])

            ext_res = get_extended_stats(ctx_path, data_h, seed=ctx["data_h_setup"]["split_seed"], sklearn_warm=sklearn_warm)
            extend_train_accs.append(ext_res[0])
            extend_test_accs.append(ext_res[2])

            if sklearn_warm:
                skl_train_accs.append(res[4])
                skl_train_leaf_accs.append(res[5])
                skl_test_accs.append(res[6])
                skl_test_leaf_accs.append(res[7])

                skl_train_leaf_soft_accs.append(s_res[5])
                skl_test_leaf_soft_accs.append(s_res[7])

                skl_red_train_accs.append(red_res[4])
                skl_red_train_leaf_accs.append(red_res[5])
                skl_red_test_accs.append(red_res[6])
                skl_red_test_leaf_accs.append(red_res[7])

                skl_red_train_leaf_soft_accs.append(red_s_res[5])
                skl_red_test_leaf_soft_accs.append(red_s_res[7])

                skl_extend_train_accs.append(ext_res[4])
                skl_extend_test_accs.append(ext_res[6])


        jobs["TrainAcc"] = train_accs
        jobs["TrainLeafAcc"] = train_leaf_accs
        jobs["TrainLeafAccSoft"] = train_leaf_soft_accs
        jobs["TestAcc"] = test_accs
        jobs["TestLeafAcc"] = test_leaf_accs
        jobs["TestLeafAccSoft"] = test_leaf_soft_accs

        jobs["ReducedTrainAcc"] = red_train_accs
        jobs["ReducedTrainLeafAcc"] = red_train_leaf_accs
        jobs["ReducedTrainLeafAccSoft"] = red_train_leaf_soft_accs
        jobs["ReducedTestAcc"] = red_test_accs
        jobs["ReducedTestLeafAcc"] = red_test_leaf_accs
        jobs["ReducedTestLeafAccSoft"] = red_test_leaf_soft_accs
        jobs["ExtendedTrainAcc"] = extend_train_accs
        jobs["ExtendedTestAcc"] = extend_test_accs

        jobs["SoftConstraint"] = soft_constr
        jobs["EmptyLeaves"] = empty_leaves

        jobs["MisassignedPoints"] = misassigned
        jobs["ObjGap"] = gap
        jobs["ObjBound"] = obj_bound
        jobs["Status"] = status
        if sklearn_warm:
            jobs["StartTrainAcc"] = skl_train_accs
            jobs["StartTrainLeafAcc"] = skl_train_leaf_accs
            jobs["StartTrainLeafAccSoft"] = skl_train_leaf_soft_accs
            jobs["StartTestAcc"] = skl_test_accs
            jobs["StartTestLeafAcc"] = skl_test_leaf_accs
            jobs["StartTestLeafAccSoft"] = skl_test_leaf_soft_accs
            jobs["StartReducedTrainAcc"] = skl_red_train_accs
            jobs["StartReducedTrainLeafAcc"] = skl_red_train_leaf_accs
            jobs["StartReducedTrainLeafAccSoft"] = skl_red_train_leaf_soft_accs
            jobs["StartReducedTestAcc"] = skl_red_test_accs
            jobs["StartReducedTestLeafAcc"] = skl_red_test_leaf_accs
            jobs["StartReducedTestLeafAccSoft"] = skl_red_test_leaf_soft_accs
            jobs["StartExtendedTrainAcc"] = skl_extend_train_accs
            jobs["StartExtendedTestAcc"] = skl_extend_test_accs

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
