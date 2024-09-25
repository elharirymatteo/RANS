__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.MFP3D_thrusters import (
    ModularFloatingPlatform,
)
from omniisaacgymenvs.robots.articulations.views.MFP3D_view import (
    ModularFloatingPlatformView,
)
from omniisaacgymenvs.tasks.MFP3D.thruster_generator import (
    VirtualPlatform,
)
from omniisaacgymenvs.tasks.MFP3D.task_factory import (
    task_factory,
)
from omniisaacgymenvs.tasks.MFP3D.penalties import (
    EnvironmentPenalties,
)
from omniisaacgymenvs.tasks.MFP3D.disturbances import (
    Disturbances,
)
from omniisaacgymenvs.tasks.common_6DoF.core import (
    quat_to_mat,
)
from omniisaacgymenvs.tasks.MFP2D_Virtual import MFP2DVirtual

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from typing import Dict, List, Tuple
from gym import spaces
import numpy as np
import wandb
import torch
import omni
import time
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class MFP3DVirtual(MFP2DVirtual):
    """
    The main class used to run tasks on the floating platform.
    Unlike other class in this repo, this class can be used to run different tasks.
    The idea being to extend it to multitask RL in the future."""

    def __init__(
        self,
        name: str,  # name of the Task
        sim_config,  # SimConfig instance for parsing cfg
        env,  # env instance of VecEnvBase or inherited class
        offset=None,  # transform offset in World
    ) -> None:
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._enable_wandb_logs = self._task_cfg["enable_wandb_log"]
        self._platform_cfg = self._task_cfg["robot"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self._device = self._cfg["sim_device"]
        self.iteration = 0
        self.step = 0

        # Split the maximum amount of thrust across all thrusters.
        self.split_thrust = self._task_cfg["robot"]["split_thrust"]

        # Collects the platform parameters
        self.dt = self._task_cfg["sim"]["dt"]
        # Collects the task parameters
        task_cfg = self._task_cfg["sub_task"]
        reward_cfg = self._task_cfg["reward"]
        penalty_cfg = self._task_cfg["penalty"]
        domain_randomization_cfg = self._task_cfg["disturbances"]
        # Instantiate the task, reward and platform
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = EnvironmentPenalties(**penalty_cfg)
        self.virtual_platform = VirtualPlatform(
            self._num_envs, self._platform_cfg, self._device
        )
        print(self.virtual_platform._max_thrusters)
        self.DR = Disturbances(
            domain_randomization_cfg,
            num_envs=self._num_envs,
            device=self._device,
        )
        self._num_observations = self.task._num_observations
        self._max_actions = self.virtual_platform._max_thrusters
        self._num_actions = self.virtual_platform._max_thrusters
        RLTask.__init__(self, name, env)
        # Instantiate the action and observations spaces
        self.set_action_and_observation_spaces()
        # Sets the initial positions of the target and platform
        self._fp_position = torch.tensor([0.0, 0.0, 0.0])
        self._default_marker_position = torch.tensor([0.0, 0.0, 0.0])
        self._marker = None
        # Preallocate tensors
        self.actions = torch.zeros(
            (self._num_envs, self._max_actions),
            device=self._device,
            dtype=torch.float32,
        )
        self.heading = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self.all_indices = torch.arange(
            self._num_envs, dtype=torch.int32, device=self._device
        )
        self.contact_state = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        # Extra info
        self.extras = {}
        self.extras_wandb = {}
        # Episode statistics
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        self.add_stats(["normed_linear_vel", "normed_angular_vel", "actions_sum"])
        return

    def set_action_and_observation_spaces(self) -> None:
        """
        Sets the action and observation spaces.
        """

        # Defines the observation space
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    np.ones(self._num_observations) * -np.Inf,
                    np.ones(self._num_observations) * np.Inf,
                ),
                "transforms": spaces.Box(low=-1, high=1, shape=(self._max_actions, 10)),
                "masks": spaces.Box(low=0, high=1, shape=(self._max_actions,)),
                "masses": spaces.Box(low=-np.inf, high=np.inf, shape=(4,)),
            }
        )

        # Defines the action space
        if self._discrete_actions == "MultiDiscrete":
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)] * self._max_actions)
        elif self._discrete_actions == "Continuous":
            pass
        elif self._discrete_actions == "Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError(
                "The requested discrete action type is not supported."
            )

    def cleanup(self) -> None:
        """
        Prepares torch buffers for RL data collection.
        """

        # prepare tensors
        self.obs_buf = {
            "state": torch.zeros(
                (self._num_envs, self._num_observations),
                device=self._device,
                dtype=torch.float,
            ),
            "transforms": torch.zeros(
                (self._num_envs, self._max_actions, 10),
                device=self._device,
                dtype=torch.float,
            ),
            "masks": torch.zeros(
                (self._num_envs, self._max_actions),
                device=self._device,
                dtype=torch.float,
            ),
            "masses": torch.zeros(
                (self._num_envs, 4),
                device=self._device,
                dtype=torch.float,
            ),
        }

        self.states_buf = torch.zeros(
            (self._num_envs, self._num_states), device=self._device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.extras = {}

    def set_up_scene(self, scene) -> None:
        """
        Sets up the USD scene inside Omniverse for the task.

        Args:
            scene (Usd.Stage): The USD stage to setup.
        """

        # Add the floating platform, and the marker
        self.get_floating_platform()
        self.get_target()

        RLTask.set_up_scene(self, scene)

        # Collects the interactive elements in the scene
        root_path = "/World/envs/.*/Modular_floating_platform"
        self._platforms = ModularFloatingPlatformView(
            prim_paths_expr=root_path,
            name="modular_floating_platform_view",
            track_contact_forces=True,
        )

        # Add views to scene
        scene.add(self._platforms)
        scene.add(self._platforms.base)
        scene.add(self._platforms.thrusters)

        # Add arrows to scene if task is go to pose
        scene, self._marker = self.task.add_visual_marker_to_scene(scene)
        return

    def get_floating_platform(self):
        """
        Adds the floating platform to the scene.
        """

        self._fp = ModularFloatingPlatform(
            prim_path=self.default_zero_env_path + "/Modular_floating_platform",
            name="modular_floating_platform",
            translation=self._fp_position,
            cfg=self._platform_cfg,
        )
        self._sim_config.apply_articulation_settings(
            "modular_floating_platform",
            get_prim_at_path(self._fp.prim_path),
            self._sim_config.parse_actor_config("modular_floating_platform"),
        )

    def update_state(self) -> None:
        """
        Updates the state of the system.
        """

        # Collects the position and orientation of the platform
        self.root_pos, self.root_quats = self._platforms.get_world_poses(clone=True)
        # Remove the offset from the different environments
        root_positions = self.root_pos - self._env_pos
        # Collects the velocity of the platform
        self.root_velocities = self._platforms.get_velocities(clone=True)
        root_velocities = self.root_velocities.clone()
        # Add noise on obs
        root_positions = self.DR.noisy_observations.add_noise_on_pos(
            root_positions, step=self.step
        )
        root_velocities = self.DR.noisy_observations.add_noise_on_vel(
            root_velocities, step=self.step
        )
        net_contact_forces = self.compute_contact_forces()
        # Compute the heading
        heading = quat_to_mat(self.root_quats)
        # Dump to state
        self.current_state = {
            "position": root_positions,
            "orientation": heading,
            "linear_velocity": root_velocities[:, :3],
            "angular_velocity": root_velocities[:, 3:],
            "net_contact_forces": net_contact_forces,
        }

    def post_reset(self):
        """
        This function implements the logic to be performed after a reset.
        """

        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._platforms.base.get_world_poses()
        self.root_velocities = self._platforms.base.get_velocities()
        self.dof_pos = self._platforms.get_joint_positions()
        self.dof_vel = self._platforms.get_joint_velocities()

        # Set initial conditions
        self.initial_root_pos, self.initial_root_rot = (
            self.root_pos.clone(),
            self.root_rot.clone(),
        )
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros(
            (self._num_envs, 4), dtype=torch.float32, device=self._device
        )
        self.initial_pin_rot[:, 0] = 1
        # Set the initial contact state
        self.contact_state = torch.zeros(
            (self._num_envs),
            dtype=torch.float32,
            device=self._device,
        )

        # control parameters
        self.thrusts = torch.zeros(
            (self._num_envs, self._max_actions, 3),
            dtype=torch.float32,
            device=self._device,
        )

        self.set_targets(self.all_indices)

    def set_targets(self, env_ids: torch.Tensor) -> None:
        """
        Sets the targets for the task.

        Args:
            env_ids: The indices of the environments to set the targets for.
        """

        num_sets = len(env_ids)
        env_long = env_ids.long()
        # Randomizes the position of the ball on the x y z axes
        target_positions, target_orientation = self.task.get_goals(
            env_long,
            step=self.step,
        )
        if len(target_positions.shape) == 3:
            position = (
                target_positions
                + self.initial_pin_pos[env_long]
                .view(num_sets, 1, 3)
                .expand(*target_positions.shape)
            ).reshape(-1, 3)
            a = (env_long * target_positions.shape[1]).repeat_interleave(
                target_positions.shape[1]
            )
            b = torch.arange(target_positions.shape[1], device=self._device).repeat(
                target_positions.shape[0]
            )
            env_long = a + b
            target_orientation = target_orientation.reshape(-1, 4)
        else:
            position = target_positions + self.initial_pin_pos[env_long]

        # Apply the new goals
        if self._marker:
            self._marker.set_world_poses(
                position,
                target_orientation,
                indices=env_long,
            )

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Resets the environments with the given indices.

        Args:
            env_ids (torch.Tensor): the indices of the environments to be reset.
        """

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        self.set_targets(env_ids)
        self.virtual_platform.randomize_thruster_state(env_ids, num_resets)
        self.DR.force_disturbances.generate_forces(env_ids, num_resets, step=self.step)
        self.DR.torque_disturbances.generate_torques(
            env_ids, num_resets, step=self.step
        )
        self.DR.mass_disturbances.randomize_masses(env_ids, step=self.step)
        # CoM_shift = self.DR.mass_disturbances.get_CoM(env_ids)
        # random_mass = self.DR.mass_disturbances.get_masses(env_ids)
        # self._platforms.base.set_masses(random_mass, indices=env_ids)
        # com_pos, com_ori = self._platforms.base.get_coms(indices=env_ids)
        # com_pos[:, 0] = CoM_shift
        # self._platforms.base.set_coms(com_pos, com_ori, indices=env_ids)
        # Randomizes the starting position of the platform
        pos, quat, vel = self.task.get_initial_conditions(env_ids, step=self.step)
        root_pos = pos + self.initial_root_pos[env_ids]
        # apply resets
        vel = torch.zeros((num_resets, 6), dtype=torch.float32, device=self._device)
        self._platforms.set_world_poses(root_pos, quat, indices=env_ids)
        self._platforms.set_velocities(vel, indices=env_ids)
        dof_pos = torch.zeros(
            (num_resets, self._platforms.num_dof), device=self._device
        )
        self._platforms.set_joint_positions(dof_pos, indices=env_ids)

        dof_vel = torch.zeros(
            (num_resets, self._platforms.num_dof), device=self._device
        )
        self._platforms.set_joint_velocities(dof_vel, indices=env_ids)
        # Resets the contacts
        self.contact_state[env_ids] = 0

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # fill `extras`
        self.extras["episode"] = {}
        self.extras_wandb = {}
        for key in self.episode_sums.keys():
            value = (
                torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
            )
            if key in self._penalties.get_stats_name():
                self.extras_wandb[key] = value
            elif key in self.task.log_with_wandb:
                self.extras_wandb[key] = value
            else:
                self.extras["episode"][key] = value
            self.episode_sums[key][env_ids] = 0.0

    def update_state_statistics(self) -> None:
        """
        Updates the statistics of the state of the training."""

        self.episode_sums["normed_linear_vel"] += torch.norm(
            self.current_state["linear_velocity"], dim=-1
        )
        self.episode_sums["normed_angular_vel"] += torch.norm(
            self.current_state["angular_velocity"], dim=-1
        )
        self.episode_sums["actions_sum"] += torch.sum(self.actions, dim=-1)
