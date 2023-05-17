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

import wandb

import os

def plot_episode_data(ep_data, save_dir):

    control_history = ep_data['act']
    reward_history = ep_data['rews']
    info_history = ep_data['info']
    obs_history = ep_data['obs']
    tgrid = np.linspace(0, len(control_history), len(control_history))
    fig_count = 0

    # °°°°°°°°°°°°°°°°°°°°°°°° plot linear speeds in time °°°°°°°°°°°°°°°°°°°°°°°°°
    lin_vels = obs_history[:, 12:15]
    # plot linear velocity
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, lin_vels[:, 0], 'r-')
    plt.plot(tgrid, lin_vels[:, 1], 'g-')
    plt.plot(tgrid, lin_vels[:, 2], 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Velocity [m/s]')
    plt.legend(['x', 'y'], loc='lower right')
    plt.title('Velocity state history')
    plt.grid()
    plt.savefig(save_dir + '_lin_vel')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot angular speeds in time °°°°°°°°°°°°°°°°°°°°°°°°°
    ang_vels = obs_history[:, 15:18]
    # plot angular speed (z coordinate)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    #plt.plot(tgrid, ang_vels[:, 0], 'r-')
    #plt.plot(tgrid, ang_vels[:, 1], 'g-')
    plt.plot(tgrid, ang_vels[:, 2], 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Angular speed [rad/s]')
    plt.legend(['z'], loc='lower right')
    plt.title('Angular speed state history')
    plt.grid()
    plt.savefig(save_dir + '_ang_vel')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot distance to target time °°°°°°°°°°°°°°°°°°°°°°°°°
    positions = obs_history[:, :3]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, positions[:, 0], 'r-')
    plt.plot(tgrid, positions[:, 1], 'g-')
    plt.xlabel('Time steps')
    plt.ylabel('Position [m]')
    plt.legend(['x', 'y'], loc='lower right')
    plt.title('Position state history')
    plt.grid()
    plt.savefig(save_dir + '_pos')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot actions in time °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = np.array(control_history)
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b-', 'g-', 'r-', 'c-']
    for k in range(control_history.shape[1]):
        col = colours[k % control_history.shape[0]]
        plt.step(tgrid, control_history[:, k], col)
    plt.xlabel('Time steps')
    plt.ylabel('Control [N]')
    plt.legend([f'u{k}' for k in range(control_history.shape[1])], loc='lower right')
    plt.title('Thrust control')
    plt.grid()
    plt.savefig(save_dir + '_actions')
    
        # °°°°°°°°°°°°°°°°°°°°°°°° plot actions histogram °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = np.array(control_history)
    n_bins = len(control_history[0])  
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator) 
    n, bins, patches = plt.hist(np.sum(control_history, axis=1), bins=len(control_history[0]), edgecolor='white')

    xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
    ticklabels = [f'T{i+1}' for i in range(n_bins)]
    plt.xticks(xticks, ticklabels)
    plt.yticks([])
    for idx, value in enumerate(n):
        if value > 0:
            plt.text(xticks[idx], value+5, int(value), ha='center')
    plt.title('Number of thrusts in episode')
    
    plt.savefig(save_dir + '_actions_hist')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot rewards °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, reward_history, 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Reward')
    plt.legend(['reward'], loc='lower right')
    plt.title('Reward history')
    plt.grid()
    # plt.show()
    plt.savefig(save_dir + '_reward')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot dist to target °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    if 'episode' in info_history[0].keys() and ('position_error' and 'heading_error') in info_history[0]['episode'].keys():
        pos_error = np.array([info_history[j]['episode']['position_error'].cpu().numpy() for j in range(len(info_history))])
        head_error = np.array([info_history[j]['episode']['heading_error'].cpu().numpy() for j in range(len(info_history))])
        plt.figure(fig_count)
        plt.clf()

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Position error [m]', color=color)
        ax1.plot(tgrid, pos_error, color=color)
        ax1.grid()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Heading error [rad]', color=color)  # we already handled the x-label with ax1
        ax2.plot(tgrid, head_error, color=color)
        plt.title('Precision metrics')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #ax2.grid()
        plt.savefig(save_dir + '_distance_error')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot dist to target °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    if 'episode' in info_history[0].keys() and ('position_reward' and 'heading_reward') in info_history[0]['episode'].keys():
        pos_error = np.array([info_history[j]['episode']['position_reward'].cpu().numpy() for j in range(len(info_history))])
        head_error = np.array([info_history[j]['episode']['heading_reward'].cpu().numpy() for j in range(len(info_history))])
        plt.figure(fig_count)
        plt.clf()

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Position reward', color=color)
        ax1.plot(tgrid, pos_error, color=color)
        ax1.grid()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Heading reward', color=color)  # we already handled the x-label with ax1
        ax2.plot(tgrid, head_error, color=color)
        plt.title('Rewards')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #ax2.grid()
        plt.savefig(save_dir + '_rewards_infos')

def eval_single_agent(cfg_dict, cfg, env):
    base_dir = "./evaluations/"
    evaluation_dir = base_dir + str(cfg_dict["task"]["name"]) + "/"
    os.makedirs(evaluation_dir, exist_ok=True)

    player = PpoPlayerDiscrete(cfg_dict['train']['params'])
    player.restore(cfg.checkpoint)
    # _____Run Evaluation_____
    num_episodes = 1 
    obs = env.reset()
    ep_data = {'act': [], 'obs': [], 'rews': [], 'info': []}
    info = {}
    for episode in range(num_episodes):
        done = 0
        step = 0
        rew_sum = torch.zeros(1, device=cfg_dict['train']['params']['config']['device'])

        while not done:
            action = player.get_action(obs['obs'], is_deterministic=True)
            #env._task.pre_physics_step(actions)
            obs, rews, dones, info =  env.step(action)
            done = dones[0]
            rew_sum += rews[0]
            step += 1
            ep_data['act'].append(action.cpu().numpy())
            ep_data['obs'].append(obs['obs'].cpu().numpy().flatten())
            ep_data['rews'].append(rew_sum.cpu().numpy())
            ep_data['info'].append(info)
        ep_data['obs'] = np.array(ep_data['obs'])
            #print(f'Step {step}: action={action}, obs={obs}, rews={rews}, dones={dones}, info={info} \n')
        print(f'Episode {episode}: rew_sum={rew_sum}info \n')
        print(ep_data)
        #plot_episode_data(ep_data, evaluation_dir)

def eval_multi_agents(cfg):
    
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    print(rlg_config_dict)
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    agent = runner.create_player()
    agent.restore(cfg.checkpoint)

    is_done = False
    env = agent.env
    obs = env.reset()
    #first_obs = {key: value[0] for key, value in obs['obs'].items()}
    #print(first_obs)

    total_reward = 0
    num_steps = 0
    while not is_done:
        actions = agent.get_action(obs, is_deterministic=True)
        obs, reward, done, info = env.step(actions)
        
        print(f'Step {num_steps}: obs={obs["obs"]}, rews={reward}, dones={done}, info={info} \n')

        total_reward += reward
        num_steps += 1
        is_done = done

    print(total_reward, num_steps)

    # runner.run({
    #     'train': False,
    #     'play': True,
    #     'checkpoint': cfg.checkpoint,
    #     'sigma': None
    # })

def activate_wandb(cfg, cfg_dict, task):
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

    # set congig params for evaluation
    cfg.task.env.maxEpisodeLength = 300
    # TODO: check how to change the following parameters for evaluation (now only possible from inside task class)
    # cfg.task.env.task_parameters['max_spawn_dist'] = 5.0
    # cfg.task.env.task_parameters['min_spawn_dist'] = 4.0   
    # cfg.task.env.task_parameters['kill_dist'] = 8.0
    # TODO: check error with visualizer of thrusters....
    #cfg.task.env.platform.configuration.visualize = False
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # _____Create environment_____

    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)

    #env.task._task_parameters.max_spawn_dist = 5.0
    
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
    eval_multi_agents(cfg)

    if cfg.wandb_activate:
        wandb.finish()
    env.close()

if __name__ == '__main__':
    parse_hydra_configs()
