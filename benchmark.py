import os
import subprocess
import pickle

from utils.datasets import DATASETS

general_config = {
    "depth": 4,
    "train_data_limit": 10_000,
    "round_limit": 5,
    "memory_limit": 250,
    "thread_limit": 8,
    "mip_focus": 0,
    "random_runs": 10,
}

# for straightforward approach
# configuration = {
#     "variant": "directMIP",
#     "base_dir": f"benchmark/direct/d4_10k",
#     "shortcut": f"D10k",
#     "script_path": "highest_acc_tester_openml.py",
#     "time_limit": 4*3600, # 4 hours
# }

# for sklearn warmstart
# configuration = {
#     "variant": "sklearn_start",
#     "base_dir": f"benchmark/warmstart/d4_10k",
#     "shortcut": f"W",
#     "script_path": "sklearn_warmstart.py",
#     "time_limit": 4*3600, # 4 hours
# }

# for gradual increase of depth
configuration = {
    "variant": "gradual_increase",
    "base_dir": f"benchmark/gradual/d4_10k_30min_first",
    "shortcut": f"G",
    "script_path": "gradual_depth_increase.py",
    "time_limit": 3600 // 2, # 30 minutes (for first, then are doubled)
}

# for halving algorithm
# configuration = {
#     "variant": "halvingMIP",
#     "base_dir": "benchmark/halving/d4_10k",
#     "shortcut": "H",
#     "script_path": "highest_acc_tester_openml.py",
#     "time_limit": 3600, # 1 hour

#     "feasibility_only": True, # faster this way
#     "halving": True,
#     "upper_limit": 1,
#     "lower_limit": 0,
#     "required_precision": 0.001,
# }


base_command = [
    "run_python_batch.script",
    configuration['script_path'],
    f"-d {general_config['depth']}",
    f"-max {general_config['train_data_limit']}",
    f"-t {configuration['time_limit']}",
    f"-m {general_config['memory_limit']}",
    f"-thr {general_config['thread_limit']}",
    f"-r {general_config['round_limit']}",
    f"-focus {general_config['mip_focus']}",
]

os.makedirs(configuration["base_dir"], exist_ok=True)
with open(configuration["base_dir"] + "/config.pickle", "wb") as f:
    pickle.dump((general_config, configuration, DATASETS), f)

jobs = []
for rand_seed in range(general_config["random_runs"]):
    for dataset_type in DATASETS:
        for dataset_name, dataset_path in DATASETS[dataset_type].items():
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
                f"--out={outfile}",
                f"--job-name={job_name}"
            ]
            result = subprocess.run(precommand + command, stdout=subprocess.PIPE, encoding='ascii')
            # writes job id to stdout that is useful for later
            jobs.append((job_name, result.stdout.strip(), outfile))

with open(configuration["base_dir"]+"/jobs", "w") as f:
    for item in jobs:
        f.write(",".join(item)+"\n")
