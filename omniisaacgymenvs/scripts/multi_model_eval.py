from json import load
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import datetime
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from rl_games.algos_torch.players import PpoPlayerDiscrete
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv

from rlgames_train import RLGTrainer
from rl_games.torch_runner import Runner
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from torch._C import fork

from utils.plot_experiment import plot_episode_data_virtual
from utils.eval_metrics import get_GoToXY_success_rate
import wandb

import os
import glob
import pandas as pd
from tqdm import tqdm

# filter out invalid experiments and retrieve valid models
def get_valid_models(load_dir, experiments):
    valid_models = []
    invalid_experiments = []
    for experiment in experiments:
        try:
            file_pattern = os.path.join(load_dir, experiment, "nn", "last_*ep_2000_rew__*.pth")
            model = glob.glob(file_pattern)
            if model:
                valid_models.append(model[0])
        except:
            invalid_experiments.append(experiment)
    if invalid_experiments:
        print(f'Invalid experiments: {invalid_experiments}')
    else:
        print('All experiments are valid')
    return valid_models


def eval_multi_agents(cfg, agent, models, horizon, load_dir):

    evaluation_dir = "./evaluations/" + load_dir
    os.makedirs(evaluation_dir, exist_ok=True)

    store_all_agents = True # store all agents generated data, if false only the first agent is stored
    is_done = False
    all_success_rate_df = pd.DataFrame()
    
    for i, model in enumerate(tqdm(models)):
        agent.restore(model)
        env = agent.env
        obs = env.reset()
        ep_data = {'act': [], 'obs': [], 'rews': [], 'all_dist': []}
        # if conf parameter kill_thrusters is true, print the thrusters that are killed for each episode 
        if cfg.task.env.platform.randomization.kill_thrusters:
            killed_thrusters_idxs = env._task.virtual_platform.action_masks

        for _ in range(horizon):
            actions = agent.get_action(obs['obs'], is_deterministic=True)
            obs, reward, done, info = env.step(actions)
        
            if store_all_agents:
                ep_data['act'].append(actions.cpu().numpy())
                ep_data['obs'].append(obs['obs']['state'].cpu().numpy())
                ep_data['rews'].append(reward.cpu().numpy())  
            else:
                ep_data['act'].append(actions[0].cpu().numpy())
                ep_data['obs'].append(obs['obs']['state'][0].cpu().numpy())
                ep_data['rews'].append(reward[0].cpu().numpy())

            is_done = done.any()
        ep_data['obs'] = np.array(ep_data['obs'])
        ep_data['rews'] = np.array(ep_data['rews'])
        ep_data['act'] = np.array(ep_data['act'])
        # if thrusters were killed during the episode, save the action with the mask applied to the thrusters that were killed
        if cfg.task.env.platform.randomization.kill_thrusters:
            ep_data['act'] = ep_data['act'] * (1 - killed_thrusters_idxs.cpu().numpy())


    # Find the episode where the sum of actions has only zeros (no action) for all the time steps
    broken_episodes = [i for i in range(0,ep_data['act'].shape[1]) if ep_data['act'][:,i,:].sum() == 0]
    # Remove episodes that are broken by the environment (IsaacGym bug)
    if broken_episodes:
        print(f'Broken episodes: {broken_episodes}')
        print(f'Ep data shape before: {ep_data["act"].shape}')
        for key in ep_data.keys():
            ep_data[key] = np.delete(ep_data[key], broken_episodes, axis=1) 
        print(f'Ep data shape after: {ep_data["act"].shape}')

        # Collect the data for the success rate table        
        success_rate_df = get_GoToXY_success_rate(ep_data, threshold=0.02)['position']
        success_rate_df['avg_rew'] = [np.mean(ep_data['rews'])]
        ang_vel_z = np.absolute(ep_data['obs'][:, :, 4:5][:,:,0])
        success_rate_df['avg_ang_vel'] = [np.mean(ang_vel_z.mean(axis=1))]
        lin_vel_x = ep_data['obs'][:, 2:3]
        lin_vel_y = ep_data['obs'][:, 3:4]
        lin_vel = np.linalg.norm(np.array([lin_vel_x, lin_vel_y]), axis=0)
        success_rate_df['avg_lin_vel'] = [np.mean(lin_vel.mean(axis=1))]

        success_rate_df['avg_action_count'] = [np.mean(np.sum(ep_data['act'], axis=1))]

        all_success_rate_df = pd.concat([all_success_rate_df, success_rate_df], ignore_index=True)
        # If want to print the latex code for the table use the following line

    # create index for the dataframe and save it
    model_names = [model.split("/")[3] for model in models]
    all_success_rate_df.insert(loc=0, column="model", value=model_names)
    all_success_rate_df.to_csv(evaluation_dir + "multi_model_performance.csv")


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    # specify the experiment load directory
    load_dir = "./models/icra24_fail/" #+ "expR_SE/"
    experiments = os.listdir(load_dir)
    print(f'Experiments found in {load_dir} folder: {len(experiments)}')
    models = get_valid_models(load_dir, experiments)
    models = [m for m in models if "BB" not in m.split("/")[3]]
    print(f'Final models: {(models)}')
    if not models:
        print('No valid models found')
        exit()
        
    # _____Create task_____
    
    # customize environment parameters based on model
    if "BB" in models[0]:
        print("Using BB model ...")
        cfg.train.params.network.mlp.units = [256, 256]
    if "BB" in models[0]:
        print("Using BBB model ...")
        cfg.train.params.network.mlp.units = [256, 256, 256]
    if "AN" in models[0]:
            print("Adding noise on act ...")
            cfg.task.env.add_noise_on_act = True
    if "AVN" in models[0]:
            print("Adding noise on act and vel ...")
            cfg.task.env.add_noise_on_act = True
            cfg.task.env.add_noise_on_vel = True
    if "UF" in cfg.checkpoint:
        print("Setting uneven floor in the environment ...")
        cfg.task.env.use_uneven_floor = True
        cfg.task.env.max_floor_force = 0.25

    horizon = 500
    cfg.task.env.maxEpisodeLength = horizon + 2
    cfg.task.env.platform.core.mass = 5.32
    cfg.task.env.split_thrust = True
    cfg.task.env.clipObservations['state'] = 20.0
    cfg.task.env.task_parameters['max_spawn_dist'] = 4.0
    cfg.task.env.task_parameters['min_spawn_dist'] = 3.0
    cfg.task.env.task_parameters['kill_dist'] = 6.0
    cfg.task.env.task_parameters['kill_after_n_steps_in_tolerance'] = 500
    
    cfg_dict = omegaconf_to_dict(cfg)
    #print_dict(cfg_dict)

    # _____Create environment_____

    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    task = initialize_task(cfg_dict, env)
    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)

    # _____Create players (model)_____
    
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    agent = runner.create_player()

    eval_multi_agents(cfg, agent, models, horizon, load_dir)

    env.close()    


if __name__ == '__main__':
    parse_hydra_configs()
