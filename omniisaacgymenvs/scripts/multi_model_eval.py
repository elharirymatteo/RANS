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
from utils.eval_metrics import success_rate_from_distances
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


def eval_multi_agents(agent, models, horizon):

    base_dir = "./evaluations/" + "icra24/" + "linR/"
    experiment_name = models[0].split("/")[1]
    print(f'Experiment name: {experiment_name}')
    evaluation_dir = base_dir + experiment_name + "/"
    os.makedirs(evaluation_dir, exist_ok=True)

    store_all_data = True # store all agents generated data, if false only the first agent is stored
    is_done = False
    all_success_rate_df = pd.DataFrame()
    
    for i, model in enumerate(tqdm(models)):
        agent.restore(model)
        env = agent.env
        obs = env.reset()
        ep_data = {'act': [], 'obs': [], 'rews': [], 'info': [], 'all_dist': []}
        
        for _ in range(horizon):
            actions = agent.get_action(obs['obs'], is_deterministic=True)
            obs, reward, done, info = env.step(actions)
            
            #print(f'Step {num_steps}: obs={obs["obs"]}, rews={reward}, dones={done}, info={info} \n')
            if store_all_data:
                ep_data['act'].append(actions.cpu().numpy())
                ep_data['obs'].append(obs['obs']['state'].cpu().numpy())
                ep_data['rews'].append(reward.cpu().numpy())  
            #ep_data['info'].append(info)
            x_pos = obs['obs']['state'][:,6].cpu().numpy()
            y_pos = obs['obs']['state'][:,7].cpu().numpy()
            ep_data['all_dist'].append(np.linalg.norm(np.array([x_pos, y_pos]), axis=0))
            is_done = done.any()

        if store_all_data:
            ep_data['obs'] = np.array(ep_data['obs'])
            ep_data['act'] = np.array(ep_data['act'])
            ep_data['rews'] = np.array(ep_data['rews'])
            
        ep_data['all_dist'] = np.array(ep_data['all_dist'])
    
        #plot_episode_data_virtual(ep_data, evaluation_dir, store_all_data)
        success_rate_df = success_rate_from_distances(ep_data['all_dist'])
        success_rate_df['avg_rew'] = [np.mean(ep_data['rews'])]
        ang_vel_z = ep_data['obs'][:, :, 4:5][:,:,0]
        success_rate_df['avg_ang_vel'] = [np.mean(ang_vel_z.mean(axis=1))]
        success_rate_df['avg_action_count'] = [np.mean(np.sum(ep_data['act'], axis=1))]
        plot_episode_data_virtual
        all_success_rate_df = pd.concat([all_success_rate_df, success_rate_df], ignore_index=True)
        # If want to print the latex code for the table use the following line
        #get_success_rate_table(success_rate_df)

    # create index for the dataframe and save it
    all_success_rate_df.index = models
    all_success_rate_df.to_csv(evaluation_dir + "multi_model_performance.csv")


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    # specify the experiment load directory
    load_dir = "./icra24/" + "linR/"
    experiments = os.listdir(load_dir)
    print(f'Experiments found in {load_dir} folder: {len(experiments)}')
    models = get_valid_models(load_dir, experiments)
    if not models:
        print('No valid models found')
        exit()
    
    # _____Create task_____
    horizon = 500
    cfg.task.env.maxEpisodeLength = horizon + 2
    cfg.task.env.platform.core.mass = 5.32
    cfg.task.env.clipObservations['state'] = 20.0
    cfg.task.env.task_parameters['max_spawn_dist'] = 3.0
    cfg.task.env.task_parameters['min_spawn_dist'] = 1.5  
    cfg.task.env.task_parameters['kill_dist'] = 6.0
    cfg.task.env.task_parameters['kill_after_n_steps_in_tolerance'] = horizon
    
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

    eval_multi_agents(agent, models, horizon)

    env.close()    


if __name__ == '__main__':
    parse_hydra_configs()
