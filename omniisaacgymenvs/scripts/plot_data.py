import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path
from utils.plot_experiment import plot_one_episode

if __name__ == "__main__":

    load_dir = Path("./lab_tests/icra24_AN/")
    sub_dirs = [d for d in load_dir.iterdir() if d.is_dir()]
    if sub_dirs:
        latest_exp = max(sub_dirs, key=os.path.getmtime)
        print("Plotting data for experiment:", latest_exp)
        n_episodes = 1
    else:
        print("No experiments found in", load_dir)
        exit()
        
    for d in sub_dirs:
        print(d)
        obs_path = os.path.join(d, "obs.npy")
        actions_path = os.path.join(d, "act.npy")
        
        if not os.path.exists(obs_path) or not os.path.exists(actions_path):
            print("Required files not found in", d)
            exit()

        obs = np.load(obs_path, allow_pickle=True)
        actions = np.load(actions_path)
        # transform the obs numpy array of dictionaries to numpy array of arrays
        obs = np.array([o['state'].cpu().numpy() for o in obs])

        save_to = os.path.join(d, 'plots/')
        os.makedirs(save_to, exist_ok=True)

        ep_data = {'act': actions, 'obs': obs, 'rews': []}

        plot_one_episode(ep_data, save_to)

    print("Done!")



