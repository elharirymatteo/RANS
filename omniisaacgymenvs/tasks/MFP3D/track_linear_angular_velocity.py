__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_6DoF.core import (
    Core,
)
from omniisaacgymenvs.tasks.MFP3D.task_rewards import (
    TrackLinearAngularVelocityReward,
)
from omniisaacgymenvs.tasks.MFP3D.task_parameters import (
    TrackLinearAngularVelocityParameters,
)
from omniisaacgymenvs.tasks.common_3DoF.curriculum_helpers import (
    CurriculumSampler,
)
from omniisaacgymenvs.tasks.MFP2D.track_linear_angular_velocity import (
    TrackLinearAngularVelocityTask as TrackLinearAngularVelocityTask2D,
)

from omni.isaac.core.prims import XFormPrimView
from pxr import Usd

from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import wandb
import torch
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class TrackLinearAngularVelocityTask(TrackLinearAngularVelocityTask2D, Core):
    """
    Implements the GoToPose task. The robot has to reach a target position and heading.
    """

    def __init__(
        self, task_param: dict, reward_param: dict, num_envs: int, device: str
    ) -> None:
        """
        Initializes the GoToPoseTask.

        Args:
            task_param (dict): The parameters of the task.
            reward_param (dict): The parameters of the reward.
            num_envs (int): The number of environments.
            device (str): The device to run the task on.
        """
        Core.__init__(self, num_envs, device)
        # Task and reward parameters
        self._task_parameters = TrackLinearAngularVelocityParameters(**task_param)
        self._reward_parameters = TrackLinearAngularVelocityReward(**reward_param)

        # Curriculum
        self._target_linear_velocity_sampler = CurriculumSampler(
            self._task_parameters.target_linear_velocity_curriculum,
        )
        self._target_angular_velocity_sampler = CurriculumSampler(
            self._task_parameters.target_angular_velocity_curriculum,
        )
        self._spawn_linear_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_linear_velocity_curriculum,
        )
        self._spawn_angular_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_angular_velocity_curriculum,
        )

        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_linear_velocities = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self._target_angular_velocities = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 3

    def update_observation_tensor(self, current_state: dict) -> torch.Tensor:
        """
        Updates the observation tensor with the current state of the robot.

        Args:
            current_state (dict): The current state of the robot.

        Returns:
            torch.Tensor: The observation tensor.
        """

        return Core.update_observation_tensor(self, current_state)

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            current_state (dict): The current state of the robot.

        Returns:
            torch.Tensor: The observation tensor.
        """

        self._linear_velocity_error = (
            self._target_linear_velocities - current_state["linear_velocity"]
        )
        self._angular_velocity_error = (
            self._target_angular_velocities - current_state["angular_velocity"]
        )
        self._position_error = current_state["position"]
        self._task_data[:, :3] = self._linear_velocity_error
        self._task_data[:, 3:6] = self._angular_velocity_error
        return self.update_observation_tensor(current_state)

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.

        Returns:
            torch.Tensor: The reward for the current state of the robot.
        """

        # position error
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        self.linear_velocity_dist = torch.sqrt(
            torch.square(self._linear_velocity_error).sum(-1)
        )
        self.angular_velocity_dist = torch.sqrt(
            torch.square(self._angular_velocity_error).sum(-1)
        )

        # Checks if the goal is reached
        lin_goal_is_reached = (
            self.linear_velocity_dist < self._task_parameters.lin_vel_tolerance
        ).int()
        ang_goal_is_reached = (
            self.angular_velocity_dist < self._task_parameters.ang_vel_tolerance
        ).int()
        goal_is_reached = lin_goal_is_reached * ang_goal_is_reached
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        # Rewards
        (
            self.linear_velocity_reward,
            self.angular_velocity_reward,
        ) = self._reward_parameters.compute_reward(
            current_state,
            actions,
            self.linear_velocity_dist,
            self.angular_velocity_dist,
        )

        return self.linear_velocity_reward + self.angular_velocity_reward

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        Generates a random goal for the task.
        Args:
            env_ids (torch.Tensor): The ids of the environments.
            target_positions (torch.Tensor): The target positions.
            target_orientations (torch.Tensor): The target orientations.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            list: The target positions and orientations.
        """

        num_goals = len(env_ids)
        # Randomizes the target linear velocity
        r = self._target_linear_velocity_sampler.sample(
            num_goals, step=step, device=self._device
        )
        theta = torch.rand((num_goals,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_goals,), device=self._device) * math.pi

        self._target_linear_velocities[env_ids, 0] = (
            r * torch.cos(theta) * torch.sin(phi)
        )
        self._target_linear_velocities[env_ids, 1] = (
            r * torch.sin(theta) * torch.sin(phi)
        )
        self._target_linear_velocities[env_ids, 2] = r * torch.cos(phi)
        # Randomizes the target angular velocity
        r = self._target_angular_velocity_sampler.sample(
            num_goals, step=step, device=self._device
        )
        theta = torch.rand((num_goals,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_goals,), device=self._device) * math.pi

        self._target_angular_velocities[env_ids, 0] = (
            r * torch.cos(theta) * torch.sin(phi)
        )
        self._target_angular_velocities[env_ids, 1] = (
            r * torch.sin(theta) * torch.sin(phi)
        )
        self._target_angular_velocities[env_ids, 2] = r * torch.cos(phi)

        # This does not matter
        return target_positions, target_orientations

    def get_initial_conditions(
        self,
        env_ids: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates the initial conditions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot.
        """

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.reset(env_ids)
        # Randomizes the starting position of the platform
        initial_position = torch.zeros(
            (num_resets, 3), device=self._device, dtype=torch.float32
        )
        # Randomizes the heading of the platform
        initial_orientation = torch.zeros(
            (num_resets, 4), device=self._device, dtype=torch.float32
        )
        uvw = torch.rand((num_resets, 3), device=self._device)
        initial_orientation[:, 0] = torch.sqrt(uvw[:, 0]) * torch.cos(
            uvw[:, 2] * 2 * math.pi
        )
        initial_orientation[:, 1] = torch.sqrt(1 - uvw[:, 0]) * torch.sin(
            uvw[:, 1] * 2 * math.pi
        )
        initial_orientation[:, 2] = torch.sqrt(1 - uvw[:, 0]) * torch.cos(
            uvw[:, 1] * 2 * math.pi
        )
        initial_orientation[:, 3] = torch.sqrt(uvw[:, 0]) * torch.sin(
            uvw[:, 2] * 2 * math.pi
        )
        # Randomizes the linear velocity of the platform
        initial_velocity = torch.zeros(
            (num_resets, 6), device=self._device, dtype=torch.float32
        )
        linear_velocity = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        initial_velocity[:, 0] = linear_velocity * torch.cos(theta) * torch.sin(phi)
        initial_velocity[:, 1] = linear_velocity * torch.sin(theta) * torch.sin(phi)
        initial_velocity[:, 2] = linear_velocity * torch.cos(phi)
        # Randomizes the angular velocity of the platform
        angular_velocity = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        initial_velocity[:, 3] = angular_velocity * torch.cos(theta) * torch.sin(phi)
        initial_velocity[:, 4] = angular_velocity * torch.sin(theta) * torch.sin(phi)
        initial_velocity[:, 5] = angular_velocity * torch.cos(phi)

        return (
            initial_position,
            initial_orientation,
            initial_velocity,
        )

    def log_spawn_data(self, step: int) -> dict:
        """
        Logs the spawn data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The spawn data.
        """
        dict = {}

        num_resets = self._num_envs
        # Randomizes the linear velocity of the platform
        linear_velocities = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )

        # Randomizes the angular velocity of the platform
        angular_velocities = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )

        linear_velocities = linear_velocities.cpu().numpy()
        angular_velocities = angular_velocities.cpu().numpy()

        fig, ax = plt.subplots(1, 2, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(linear_velocities, bins=32)
        ax[0].set_title("Initial normed linear velocity")
        ax[0].set_xlim(
            self._spawn_linear_velocity_sampler.get_min_bound(),
            self._spawn_linear_velocity_sampler.get_max_bound(),
        )
        ax[0].set_xlabel("vel (m/s)")
        ax[0].set_ylabel("count")
        ax[1].hist(angular_velocities, bins=32)
        ax[1].set_title("Initial normed angular velocity")
        ax[1].set_xlim(
            self._spawn_angular_velocity_sampler.get_min_bound(),
            self._spawn_angular_velocity_sampler.get_max_bound(),
        )
        ax[1].set_xlabel("vel (rad/s)")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_velocities"] = wandb.Image(data)
        return dict

    def log_target_data(self, step: int) -> dict:
        """
        Logs the target data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The target data.
        """

        dict = {}

        num_resets = self._num_envs
        # Randomizes the target linear velocity of the platform
        r = self._target_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        # Randomizes the target angular velocity
        d = self._target_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )

        r = r.cpu().numpy()
        d = d.cpu().numpy()

        fig, ax = plt.subplots(1, 2, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(r, bins=32)
        ax[0].set_title("Target normed linear velocity")
        ax[0].set_xlim(
            self._target_linear_velocity_sampler.get_min_bound(),
            self._target_linear_velocity_sampler.get_max_bound(),
        )
        ax[0].set_xlabel("vel (m/s)")
        ax[0].set_ylabel("count")
        ax[1].hist(d, bins=32)
        ax[1].set_title("Target normed angular velocity")
        ax[1].set_xlim(
            self._target_angular_velocity_sampler.get_min_bound(),
            self._target_angular_velocity_sampler.get_max_bound(),
        )
        ax[1].set_xlabel("vel (rad/s)")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/target_velocities"] = wandb.Image(data)
        return dict
