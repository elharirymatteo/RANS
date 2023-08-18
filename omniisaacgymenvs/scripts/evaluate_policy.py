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
from utils.eval_metrics import get_GoToXY_success_rate, get_GoToPose_success_rate, get_TrackXYVelocity_success_rate, get_TrackXYOVelocity_success_rate
import wandb

import os


def eval_multi_agents(cfg, horizon):
    """
    Evaluate a trained agent for a given number of steps"""

    base_dir = "./evaluations/" + cfg.checkpoint.split("/")[0] + "/" +  cfg.checkpoint.split("/")[1] + "/"
    experiment_name = cfg.checkpoint.split("/")[2]
    print(f'Experiment name: {experiment_name}')
    evaluation_dir = base_dir + experiment_name + "/"
    os.makedirs(evaluation_dir, exist_ok=True)

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    agent = runner.create_player()
    agent.restore(cfg.checkpoint)

    store_all_agents = True # store all agents generated data, if false only the first agent is stored
    is_done = False
    env = agent.env
    obs = env.reset()

    ep_data = {'act': [], 'obs': [], 'rews': []}
    total_reward = 0
    num_steps = 0
    
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
        total_reward += reward[0]
        num_steps += 1
        is_done = done.any()
    ep_data['obs'] = np.array(ep_data['obs'])
    ep_data['act'] = np.array(ep_data['act'])
    ep_data['rews'] = np.array(ep_data['rews'])

    # Find the episode where the sum of actions has only zeros (no action) for all the time steps
    broken_episodes = [i for i in range(0,ep_data['act'].shape[1]) if ep_data['act'][:,i,:].sum() == 0]
    # Remove episodes that are broken by the environment (IsaacGym bug)
    if broken_episodes:
        print(f'Broken episodes: {broken_episodes}')
        print(f'Ep data shape before: {ep_data["act"].shape}')
        for key in ep_data.keys():
            ep_data[key] = np.delete(ep_data[key], broken_episodes, axis=1) 
        print(f'Ep data shape after: {ep_data["act"].shape}')


    print(f'\n Episode: rew_sum={total_reward:.2f}, tot_steps={num_steps} \n')
    print(f'Episode data obs shape: {ep_data["obs"].shape} \n')

    if cfg.headless:
        plot_episode_data_virtual(ep_data, evaluation_dir, store_all_agents)
        task_flag = ep_data['obs'][0, 0, 5].astype(int)
        if task_flag == 0: # GoToXY
            success_rate = get_GoToXY_success_rate(ep_data, print_intermediate=True)
        elif task_flag == 1: # GoToPose
            success_rate = get_GoToPose_success_rate(ep_data, print_intermediate=True)
        elif task_flag == 2: # TrackXYVelocity
            success_rate = get_TrackXYVelocity_success_rate(ep_data, print_intermediate=True)
        elif task_flag == 3: # TrackXYOVelocity
            success_rate = get_TrackXYOVelocity_success_rate(ep_data, print_intermediate=True)

def activate_wandb(cfg, cfg_dict, task):
    """
    Activate wandb logging for the evaluation run"""

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"evaluate_{task.name}_{time_str}"
    cfg.wandb_entity = "matteohariry"
    wandb.tensorboard.patch(root_logdir="evaluations")
    wandb.init(
        project=task.name,
        entity=cfg.wandb_entity,
        config=cfg_dict,
        sync_tensorboard=True,
        name=run_name,
        resume="allow",
        group="eval_agents"
    )
    
@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    if cfg.checkpoint is None:
        print("No checkpoint specified. Exiting...")
        return

    horizon = 500

    # choose frequency of evaluation (to be multiplied by 10 -> controlFrequencyInv)
    # set congig params for evaluation
    cfg.task.env.maxEpisodeLength = horizon + 2
    
    cfg.task.env.split_thrust = True
    cfg.task.env.task_parameters['kill_after_n_steps_in_tolerance'] = 500
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # _____Create environment_____

    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)

    
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    # Activate wandb logging
    if cfg.wandb_activate:
        activate_wandb(cfg, cfg_dict, task)

    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    # _____Create players (model)_____
    
    #eval_single_agent(cfg_dict, cfg, env)
    eval_multi_agents(cfg,horizon)

    if cfg.wandb_activate:
        wandb.finish()
    env.close()

if __name__ == '__main__':
    parse_hydra_configs()