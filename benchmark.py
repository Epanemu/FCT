import os
import subprocess
import pickle

from utils.datasets import DATASETS_INFO

general_config = {
    "depth": 4,
    "min_in_leaf": 50,
    "train_data_limit": 10_000,
    "round_limit": 4,
    "memory_limit": 250,
    "thread_limit": 8,
    "time_limit": 8*3600,
    "mip_focus": 1,
    "mip_heuristics": 0.8,
    "random_runs": 10,
}

# for Direct
# configuration = {
#     "variant": "direct",
#     "base_dir": f"benchmark/direct/runname",
#     "shortcut": f"D",
#     "script_path": "direct.py",
#     "params": [],
# }

# for Warmstarted using CART
configuration = {
    "variant": "sklearn_start",
    "base_dir": f"benchmark/warmstart/runname",
    "shortcut": f"W",
    "script_path": "sklearn_warmstart.py",
    "params": ["-init hint"],
}

# for Gradual increase of depth
# configuration = {
#     "variant": "gradual_increase",
#     "base_dir": f"benchmark/gradual/runname",
#     "shortcut": f"G",
#     "script_path": "gradual_depth_increase.py",
#     "params": ["-init hint"],
# }

# for OCT direct
# configuration = {
#     "variant": "OCT",
#     "base_dir": f"benchmark/OCT/runname",
#     "shortcut": f"O",
#     "script_path": "oct.py",
#     "params": [],
# }

# for OCT direct
# configuration = {
#     "variant": "OCT",
#     "base_dir": f"benchmark/OCT/warm_runname",
#     "shortcut": f"Ow",
#     "script_path": "oct.py",
#     "params": ["-warm"],
# }

# for halving - not presented in the paper
# configuration = {
#     "variant": "halving",
#     "base_dir": f"benchmark/halving/runname",
#     "shortcut": f"H",
#     "script_path": "halving.py",
#     "params": ["-u 1 -l 0.5 -prec 0.001"], # all are binary classifications, otherwise should the lower bound be 1/K
# }


base_command = [
    "run_python_batch.script",
    configuration['script_path'],
    f"-d {general_config['depth']}",
    f"-max {general_config['train_data_limit']}",
    f"-t {general_config['time_limit']}",
    f"-m {general_config['memory_limit']}",
    f"-thr {general_config['thread_limit']}",
    f"-r {general_config['round_limit']}",
    f"-focus {general_config['mip_focus']}",
    f"-heur {general_config['mip_heuristics']}",
    f"-lmin {general_config['min_in_leaf']}",
] + configuration["params"]

os.makedirs(configuration["base_dir"], exist_ok=True)
with open(configuration["base_dir"] + "/config.pickle", "wb") as f:
    pickle.dump((general_config, configuration, DATASETS_INFO), f)

jobs = []
for rand_seed in range(general_config["random_runs"]):
    for dataset_type in DATASETS_INFO:
        for dataset_name, dataset_info in DATASETS_INFO[dataset_type].items():
# for rand_seed, dataset_type, dataset_name in [ # Uncomment for selective extra runs
#         (0, "categorical", "albert"),
#     ]:
#     if True:
#         if True:
#             dataset_info = DATASETS_INFO[dataset_type][dataset_name]
            dataset_path = dataset_info["path"]
            if dataset_info["n_features"] > 30 or (dataset_info["n_points"] > 10000/0.8 and dataset_info["n_features"] > 20):
                max_memory = 128

            res_path = os.path.join(configuration["base_dir"], dataset_type, dataset_name)
            os.makedirs(res_path, exist_ok=True)

            command = base_command + [
                f"--dataset_path {dataset_path}",
                f"--dataset_type {dataset_type}",
                f"--results_dir {res_path}",
                f"-seed {rand_seed}",
            ]
            # call to cluster manager
            job_name = f"{configuration['shortcut']}_{rand_seed}_{dataset_type[0]}_{dataset_name}"
            outfile = f"{res_path}/run{rand_seed}.out"
            precommand = [
                "sbatch",
                "--parsable",
                f"--mem={max_memory*1024}",
                f"--out={outfile}",
                f"--job-name={job_name}",
                f"--cpus-per-task={general_config['thread_limit']}",
            ]
            result = subprocess.run(precommand + command, stdout=subprocess.PIPE, encoding='ascii')
            # writes job id to stdout that is useful for later
            jobs.append((job_name, result.stdout.strip(), outfile))

with open(configuration["base_dir"]+"/jobs", "w") as f:
    for item in jobs:
        f.write(",".join(item)+"\n")
