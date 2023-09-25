from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.MFP3D_virtual_thrusters import ModularFloatingPlatform
from omniisaacgymenvs.robots.articulations.views.mfp3d_virtual_thrusters_view import ModularFloatingPlatformView

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_thruster_generator import VirtualPlatform
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_factory import task_factory
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_core import parse_data_dict, quat_to_mat
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_rewards import Penalties
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_disturbances import UnevenFloorDisturbance, TorqueDisturbance, NoisyObservations, NoisyActions

from omniisaacgymenvs.tasks.MFP2D_Virtual import MFP2DVirtual

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import omni
import time
import math
import torch
from gym import spaces
from dataclasses import dataclass

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class MFP3DVirtual(MFP2DVirtual):
    """
    The main class used to run tasks on the floating platform.
    Unlike other class in this repo, this class can be used to run different tasks.
    The idea being to extend it to multitask RL in the future."""

    def __init__(
        self,
        name: str,                # name of the Task
        sim_config,    # SimConfig instance for parsing cfg
        env,          # env instance of VecEnvBase or inherited class
        offset=None               # transform offset in World
    ) -> None:
         
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._platform_cfg = self._task_cfg["env"]["platform"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self._device = self._cfg["sim_device"]
        self.step = 0

        # Split the maximum amount of thrust across all thrusters.
        self.split_thrust = self._task_cfg['env']['split_thrust']

        # Domain randomization and adaptation
        self.UF = UnevenFloorDisturbance(self._task_cfg, self._num_envs, self._device)
        self.TD = TorqueDisturbance(self._task_cfg, self._num_envs, self._device)
        self.ON = NoisyObservations(self._task_cfg)
        self.AN = NoisyActions(self._task_cfg)
        # Collects the platform parameters
        self.dt = self._task_cfg["sim"]["dt"]
        # Collects the task parameters
        task_cfg = self._task_cfg["env"]["task_parameters"]
        reward_cfg = self._task_cfg["env"]["reward_parameters"]
        penalty_cfg = self._task_cfg["env"]["penalties_parameters"]
        # Instantiate the task, reward and platform
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = parse_data_dict(Penalties(), penalty_cfg)
        self.virtual_platform = VirtualPlatform(self._num_envs, self._platform_cfg, self._device)
        self._num_observations = self.task._num_observations
        self._max_actions = self.virtual_platform._max_thrusters
        self._num_actions = self.virtual_platform._max_thrusters
        RLTask.__init__(self, name, env)
        # Instantiate the action and observations spaces
        self.set_action_and_observation_spaces()
        # Sets the initial positions of the target and platform
        self._fp_position = torch.tensor([0.0, 0.0, 0.5])
        self._default_marker_position = torch.tensor([0.0, 0.0, 0.0])
        self._marker = None
        # Preallocate tensors
        self.actions = torch.zeros((self._num_envs, self._max_actions), device=self._device, dtype=torch.float32)
        self.heading = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        # Extra info
        self.extras = {}
        # Episode statistics
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        self.add_stats(['normed_linear_vel', 'normed_angular_vel', 'actions_sum'])
        return
    
    def set_action_and_observation_spaces(self) -> None:
        """
        Sets the action and observation spaces."""

        # Defines the observation space
        self.observation_space = spaces.Dict({"state":spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf),
                                              "transforms":spaces.Box(low=-1, high=1, shape=(self._max_actions, 10)),
                                              "masks":spaces.Box(low=0, high=1, shape=(self._max_actions,))})

        # Defines the action space
        if self._discrete_actions=="MultiDiscrete":    
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)]*self._max_actions)
        elif self._discrete_actions=="Continuous":
            pass
        elif self._discrete_actions=="Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError("The requested discrete action type is not supported.")

    def set_up_scene(self, scene) -> None:
        """
        Sets up the USD scene inside Omniverse for the task.
        
        Args:
            scene: The USD stage to setup."""

        # Add the floating platform, and the marker
        self.get_floating_platform()
        self.get_target()
        
        RLTask.set_up_scene(self, scene) 

        # Collects the interactive elements in the scene
        root_path = "/World/envs/.*/Modular_floating_platform" 
        self._platforms = ModularFloatingPlatformView(prim_paths_expr=root_path, name="modular_floating_platform_view") 

        # Add views to scene
        scene.add(self._platforms)
        scene.add(self._platforms.base)

        scene.add(self._platforms.thrusters)

        # Add arrows to scene if task is go to pose
        scene, self._marker = self.task.add_visual_marker_to_scene(scene)
        return
    
    def cleanup(self) -> None:
        """
        Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = {'state':torch.zeros((self._num_envs, self._num_observations), device=self._device, dtype=torch.float),
                        'transforms':torch.zeros((self._num_envs, self._max_actions, 10), device=self._device, dtype=torch.float),
                        'masks':torch.zeros((self._num_envs, self._max_actions), device=self._device, dtype=torch.float)}

        self.states_buf = torch.zeros((self._num_envs, self.num_states), device=self._device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.extras = {}

    def get_floating_platform(self):
        """
        Adds the floating platform to the scene."""

        fp = ModularFloatingPlatform(prim_path=self.default_zero_env_path + "/Modular_floating_platform", name="modular_floating_platform",
                            translation=self._fp_position, cfg=self._platform_cfg)
        self._sim_config.apply_articulation_settings("modular_floating_platform", get_prim_at_path(fp.prim_path),
                                                        self._sim_config.parse_actor_config("modular_floating_platform"))

    def update_state(self) -> None:
        """
        Updates the state of the system."""

        # Collects the position and orientation of the platform
        self.root_pos, self.root_quats = self._platforms.get_world_poses(clone=True)
        # Remove the offset from the different environments
        root_positions = self.root_pos - self._env_pos
        # Collects the velocity of the platform
        self.root_velocities = self._platforms.get_velocities(clone=True)
        root_velocities = self.root_velocities.clone()
        # Cast quaternion to Yaw
        # Add noise on obs
        root_positions = self.ON.add_noise_on_pos(root_positions)
        root_velocities = self.ON.add_noise_on_vel(root_velocities)
        # Compute the heading
        heading = quat_to_mat(self.root_quats)
        # Dump to state
        self.current_state = {"position":root_positions, "orientation": heading, "linear_velocity": root_velocities[:,:3], "angular_velocity":root_velocities[:,3:]}

    def set_targets(self, env_ids: torch.Tensor) -> None:
        """
        Sets the targets for the task.
        
        Args:
            env_ids: The indices of the environments to set the targets for."""

        env_long = env_ids.long()
        # Randomizes the position of the ball on the x y axis
        target_positions, target_orientation = self.task.get_goals(env_long, self.initial_pin_pos.clone(), self.initial_pin_rot.clone())
        # Apply the new goals
        if self._marker:
            self._marker.set_world_poses(target_positions[env_long], target_orientation[env_long], indices=env_long)

    def update_state_statistics(self) -> None:
        """
        Updates the statistics of the state of the training."""

        self.episode_sums['normed_linear_vel'] += torch.norm(self.current_state["linear_velocity"], dim=-1)
        self.episode_sums['normed_angular_vel'] += torch.norm(self.current_state["angular_velocity"], dim=-1)
        self.episode_sums['actions_sum'] += torch.sum(self.actions, dim=-1)