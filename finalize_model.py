# script that extends the original model by adding XGBoost to their leaves
from fct_nn.DataHandler import DataHandler
from fct_nn.FCT_Extended import FCT_Extended
from fct_nn.TreeGenerator import TreeGenerator
import pickle
import os
import sys

N_SEARCH_ITERS = 50 # 100 # do a half of them, not as many points in leaves, and want it to be reasonably fast

def make_extended_model(ctx_path, extended_path):
    print("starting model", ctx_path)
    with open(ctx_path, "rb") as f:
        ctx = pickle.load(f)
    dh_setup = ctx["data_h_setup"]
    data_h = DataHandler(dh_setup["path"], dh_setup["round_limit"])
    data_h.get_training_data(dh_setup["split_seed"], dh_setup["test_size"], dh_setup["limit"], reset_stats=True)

    seed = ctx["data_h_setup"]["split_seed"]

    gen = TreeGenerator(data_h)
    tree = gen.make_from_context(ctx)
    ext_tree = FCT_Extended(tree, data_h, seed=seed, search_iterations=N_SEARCH_ITERS)

    with open(extended_path, "wb") as f:
        pickle.dump(ext_tree.get_context(), f)
    print("wrote to", extended_path, flush=True)

ctx_path = sys.argv[1]
extended_path = ctx_path[:-4] + "_ext.ctx"
make_extended_model(ctx_path, extended_path)
