__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from utils.plot_experiment import plot_one_episode
import argparse

if __name__ == "__main__":
    # Get load dir from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load_dir", type=str, default=None, help="Directory to load data from"
    )
    args = parser.parse_args()
    load_dir = Path(args.load_dir)

    # load_dir = Path("./ros_lab_exp/7_9_23/dc_controller")
    sub_dirs = [d for d in load_dir.iterdir() if d.is_dir()]
    # sub_dirs = [d for d in sub_dirs if ("pose" not in str(d) and "kill3" not in str(d) and "new_pose" not in str(d))]
    if sub_dirs:
        latest_exp = max(sub_dirs, key=os.path.getmtime)
        n_episodes = 1
    else:
        print("No experiments found in", load_dir)
        exit()

    for d in sub_dirs:
        obs_path = os.path.join(d, "obs.npy")
        actions_path = os.path.join(d, "act.npy")

        if not os.path.exists(obs_path) or not os.path.exists(actions_path):
            print("Required files not found in", d)
            exit()

        obs = np.load(obs_path, allow_pickle=True)
        actions = np.load(actions_path)

        # if obs is empty, skip this experiment and print warning
        if not obs.any():
            print(f"Empty obs file in {d}, skipping...")
            continue

        print("Plotting data for experiment:", d)
        # transform the obs numpy array of dictionaries to numpy array of arrays
        obs = np.array([o.flatten() for o in obs])

        save_to = os.path.join(d, "plots/")
        os.makedirs(save_to, exist_ok=True)
        ep_data = {"act": actions, "obs": obs, "rews": []}

        plot_one_episode(ep_data, save_to, show=False)

    print("Done!")
