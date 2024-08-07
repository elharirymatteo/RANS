# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import gym
import hydra
from omegaconf import DictConfig
import os
import time

import numpy as np
import torch

import omniisaacgymenvs
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import get_experience
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task


@hydra.main(version_base=None, config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    headless = cfg.headless
    render = not headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras

    # select kit app file
    experience = get_experience(
        headless,
        cfg.enable_livestream,
        enable_viewport,
        cfg.enable_recording,
        cfg.kit_app,
    )

    env = VecEnvRLGames(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport or cfg.enable_recording,
        experience=experience,
    )
    # parse experiment directory
    module_path = os.path.abspath(
        os.path.join(os.path.dirname(omniisaacgymenvs.__file__))
    )
    experiment_dir = os.path.join(module_path, "runs", cfg.train.params.config.name)

    # use gym RecordVideo wrapper for viewport recording
    if cfg.enable_recording:
        if cfg.recording_dir == "":
            videos_dir = os.path.join(experiment_dir, "videos")
        else:
            videos_dir = cfg.recording_dir
        video_interval = lambda step: step % cfg.recording_interval == 0
        video_length = cfg.recording_length
        env.is_vector_env = True
        if env.metadata is None:
            env.metadata = {
                "render_modes": ["rgb_array"],
                "render_fps": cfg.recording_fps,
            }
        else:
            env.metadata["render_modes"] = ["rgb_array"]
            env.metadata["render_fps"] = cfg.recording_fps
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=videos_dir,
            step_trigger=video_interval,
            video_length=video_length,
        )

    # sets seed. if seed is -1 will pick a random one
    from omni.isaac.core.utils.torch.maths import set_seed

    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict["seed"] = cfg.seed
    task = initialize_task(cfg_dict, env)

    num_frames = 0
    first_frame = True
    prev_time = time.time()
    while env.simulation_app.is_running():
        if env.world.is_playing():
            if first_frame:
                env.reset()
                prev_time = time.time()
                first_frame = False

            # get upper and lower bounds of action space, sample actions randomly on this interval
            action_high = env.action_space.high[0]
            action_low = env.action_space.low[0]
            actions = (action_high - action_low) * torch.rand(
                env.num_envs, env.action_space.shape[0], device=task.rl_device
            ) - action_high

            if time.time() - prev_time >= 1:
                print("FPS:", num_frames, "FPS * num_envs:", env.num_envs * num_frames)
                num_frames = 0
                prev_time = time.time()
            else:
                num_frames += 1

            env.step(actions)
        else:
            env.world.step(render=render)

    env.simulation_app.close()


if __name__ == "__main__":
    parse_hydra_configs()
