import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import os
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from rl_games.algos_torch.players import PpoPlayerDiscrete
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv

from rlgames_train import RLGTrainer
from rl_games.torch_runner import Runner
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path

def eval_single_agent(cfg_dict, cfg, env):

    player = PpoPlayerDiscrete(cfg_dict['train']['params'])
    player.restore(cfg.checkpoint)
    # _____Run Evaluation_____
    num_episodes = 1 
    obs = env.reset()

    for _ in range(num_episodes):
        done = False
        while not done:
            actions = player.get_action(obs['obs'], is_deterministic=True)
            #env._task.pre_physics_step(actions)
            obs, rews, dones, _ =  env.step(actions)
            done = dones[0]
            print(f'Step {env.sim_frame_count} -- obs: {obs}, rews: {rews}, dones: {dones}')


def eval_multi_agents(cfg_dict, cfg, env):
    
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    runner.run({
        'train': False,
        'play': True,
        'checkpoint': cfg.checkpoint,
        'sigma': None
    })


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    cfg.checkpoint = "./runs/MFP2DGoToPose/nn/MFP2DGoToPose.pth"
    cfg.num_envs = 1
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
    
    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    # _____Create players (model)_____
    
    eval_single_agent(cfg_dict, cfg, env)

    env.close()

if __name__ == '__main__':
    parse_hydra_configs()
