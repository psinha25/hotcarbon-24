# Sample command:
# python3 new_stats.py \
#    --mode mps-uncap \
#    --result_dir dir1 \
#    /tmp/111003.pkl /tmp/111004.pkl /tmp/111005.pkl

import argparse
import os
import sys
import pickle
import atexit
import fcntl
import numpy as np
import pandas as pd


TPUT = "tput"
TOTAL_PREFIX = "total"
PERCENTILES = [0, 50, 90, 99, 100]

METRIC_NAMES = [TPUT]
for prefix in [TOTAL_PREFIX]:
    for percentile in PERCENTILES:
        METRIC_NAMES.append(f"{prefix}_p{percentile}")


def acquire_lock():
    print("Attempting to acquire stat lock")
    global lock_fd
    lock_fd = open(sys.argv[0], 'r+')
    fcntl.flock(lock_fd, fcntl.LOCK_EX)  # Acquire an exclusive lock
    print("Acquired stat lock")


def release_lock():
    fcntl.flock(lock_fd, fcntl.LOCK_UN)  # Release the lock
    lock_fd.close()


def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def populate_stats(id, array, tid, metrics):
    # Calculate percentiles
    percentile_metrics = np.percentile(array, PERCENTILES)
    for i, percentile in enumerate(PERCENTILES):
        metrics[f"{id}_p{percentile}"][tid] = percentile_metrics[i] * 1000


def create_dir(directory):
    try:
        os.makedirs(directory)
    except:
        pass


if __name__ == "__main__":
    acquire_lock()
    atexit.register(release_lock)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)
    parser.add_argument(
        "pickle_files",
        metavar="<pkl_file>",
        type=str,
        nargs="+",
        help="List of model and details"
    )
    opt = parser.parse_args()

    create_dir(opt.result_dir)

    models = [None] * len(opt.pickle_files)
    metrics = {}
    for metric_name in METRIC_NAMES:
        metrics[metric_name] = [None] * len(opt.pickle_files)

    for pickle_file in opt.pickle_files:
        # Validate file paths
        if not os.path.isfile(pickle_file):
            print(f"File '{pickle_file}' does not exist.")
            sys.exit(1)

        # Load arrays from pickle files
        tid, infer_stats = load_pickle_file(pickle_file)
        model, tput, total_times = infer_stats
        populate_stats(TOTAL_PREFIX, total_times, tid, metrics)
        metrics[TPUT][tid] = tput
        if models[tid] is None:
            models[tid] = f"{tid}_{model}"
        else:
            assert models[tid] == f"{tid}_{model}"

    # Create a DataFrame for each metric type
    models = [x for x in models if x is not None]
    for metric_type, metrics_list in metrics.items():
        df_data = {
            model_name: metrics_list[i] for i, model_name in enumerate(models)
        }
        df_data["mode"] = opt.mode
        df_data["load"] = 1.0
        df = pd.DataFrame(df_data, index=[metric_type])
        cols = (["mode", "load"] +
                [col for col in df.columns if col != "mode" and col != "load"])
        df = df[cols]

        csv_file = os.path.join(opt.result_dir, f"{metric_type}.csv")
        if os.path.exists(csv_file):
            df.to_csv(csv_file, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_file, index=False)