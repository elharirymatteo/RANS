__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import os
import datetime
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver

from rlgames_train import RLGTrainer
from rl_games.torch_runner import Runner
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames_mfp import VecEnvRLGames


def eval_multi_agents(cfg, horizon):
    """
    Evaluate a trained agent for a given number of steps"""
    evaluation_dir = "./sdg/" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(evaluation_dir, exist_ok=True)

    rlg_config_dict = omegaconf_to_dict(cfg.train)
    runner = Runner(RLGPUAlgoObserver())
    runner.load(rlg_config_dict)
    runner.reset()

    agent = runner.create_player()
    agent.restore(cfg.checkpoint)
    agent.has_batch_dimension = True
    agent.batch_size = 1024
    # agent.init_rnn()
    
    env = agent.env
    obs = env.reset()
    # if conf parameter kill_thrusters is true, print the thrusters that are killed for each episode
    if cfg.task.env.platform.randomization.kill_thrusters:
        killed_thrusters_idxs = env._task.virtual_platform.action_masks

    ep_data = {"act": [], "state": [], "rgb": [], "depth": [], "rews": []}
    total_reward = 0
    num_steps = 0

    for _ in range(horizon):
        actions = agent.get_action(obs["obs"], is_deterministic=True)
        obs, reward, done, info = env.step(actions)
        rgb, depth = env._task.get_rgbd_data()
        
        ep_data["act"].append(actions.cpu())
        ep_data["state"].append(obs["obs"]["state"].cpu())
        ep_data["rgb"].append(rgb.cpu())
        ep_data["depth"].append(depth.cpu())
        ep_data["rews"].append(reward.cpu())
        total_reward += reward[0]
        num_steps += 1
    ep_data["state"] = torch.cat(ep_data["state"])
    ep_data["rews"] = torch.cat(ep_data["rews"])
    ep_data["act"] = torch.cat(ep_data["act"])
    ep_data["rgb"] = torch.cat(ep_data["rgb"])
    ep_data["depth"] = torch.cat(ep_data["depth"])
    # if thrusters were killed during the episode, save the action with the mask applied to the thrusters that were killed
    if cfg.task.env.platform.randomization.kill_thrusters:
        ep_data["act"] = ep_data["act"] * (1 - killed_thrusters_idxs.cpu().numpy())

    print(f"\n Episode: rew_sum={total_reward:.2f}, tot_steps={num_steps} \n")
    print(f'Episode data numberes: {ep_data["state"].shape[0]} \n')
    
    # save the episode data
    torch.save(ep_data["state"], os.path.join(evaluation_dir, "state.pt"))
    torch.save(ep_data["act"], os.path.join(evaluation_dir, "act.pt"))
    torch.save(ep_data["rews"], os.path.join(evaluation_dir, "rews.pt"))
    torch.save(ep_data["rgb"], os.path.join(evaluation_dir, "rgb.pt"))
    torch.save(ep_data["depth"], os.path.join(evaluation_dir, "depth.pt"))


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    if cfg.checkpoint is None:
        print("No checkpoint specified. Exiting...")
        return

    horizon = 100
    cfg.task.env.maxEpisodeLength = horizon + 2
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # _____Create environment_____

    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    env = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport,
    )

    from omni.isaac.core.utils.torch.maths import set_seed

    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed
    task = initialize_task(cfg_dict, env)

    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    # _____Create players (model)_____

    # eval_single_agent(cfg_dict, cfg, env)
    eval_multi_agents(cfg, horizon)

    env.close()


if __name__ == "__main__":
    parse_hydra_configs()