import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import rl_games
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from rl_games.algos_torch.model_builder import ModelBuilder
from rl_games.common.algo_observer import DefaultAlgoObserver

import yaml
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    #print_dict(cfg_dict)
    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # Specify the testing environment config here
    cfg.task.env.envSpacing = 25


    # Load the environment & initialize the task
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)

    # Load the model
    #model_path = "./runs/FloatingPlatform/nn/FloatingPlatform.pth"
    config_path = "./runs/FloatingPlatform/config.yaml"
    with open(config_path) as f:
        model_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    params = model_cfg['train']['params']
    params['config']['features'] = {}
    params['config']['features']['observer'] = DefaultAlgoObserver()
    print(f'Len action space: {len(env.action_space)}')
    model_builder = ModelBuilder()
    model = model_builder.load(params)
    # print(env.action_space, env.observation_space, env.num_envs)
    build_config = {
        'actions_num' : np.array(len(env.action_space)),
        'input_shape' : (len(env.observation_space.sample()), env.num_envs),
        'num_seqs' : 1,
        'value_size': 1,
        'normalize_value' : False,
        'normalize_input': False,
    }
    model = model.build(build_config)
    print(model)

    n_episodes = 2
    n_steps = 100

    for _ in range(n_episodes):
        obs = env.reset()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : None
        }
        res = model(obs)
        print(f'res: {res}')
        # for i in range(n_steps):
        #     actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
        #     print(f'actions: {actions}')
        #     obs, reward, done, info = env.step(actions)
        #     print(f' step: {i} : {obs, reward, done, info}')

    env._simulation_app.close()

    # while env._simulation_app.is_running():
    #     if env._world.is_playing():
    #         if env._world.current_time_step_index == 0:
    #             env._world.reset(soft=True)
    #         actions = torch.tensor(np.array([env.action_space.sample() for _ in range(env.num_envs)]), device=task.rl_device)
    #         env._task.pre_physics_step(actions)
    #         env._world.step(render=render)
    #         env.sim_frame_count += 1
    #         env._task.post_physics_step()
    #     else:
    #         env._world.step(render=render)

    # env._simulation_app.close()

if __name__ == '__main__':
    parse_hydra_configs()

# Define evaluation criteria
def evaluate_agent(agent, n_episodes):
    env = rl_games.TicTacToeEnv()
    wins = 0
    for i in range(n_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
        if info['winner'] == 1:
            wins += 1
    win_rate = wins / n_episodes
    return win_rate

# # Load trained agent
# agent = rl_games.load_agent('path/to/agent')

# # Evaluate agent
# win_rate = evaluate_agent(agent, 100)
# print(f"Win rate: {win_rate}")
