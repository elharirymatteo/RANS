__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_6DoF.core import (
    Core,
)
from omniisaacgymenvs.tasks.MFP3D.task_rewards import (
    TrackLinearVelocityReward,
)
from omniisaacgymenvs.tasks.MFP3D.task_parameters import (
    TrackLinearVelocityParameters,
)
from omniisaacgymenvs.tasks.common_3DoF.curriculum_helpers import (
    CurriculumSampler,
)

from omniisaacgymenvs.tasks.common_3DoF.track_linear_velocity import (
    TrackLinearVelocityTask as TrackLinearVelocityTask2D,
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


class TrackLinearVelocityTask(TrackLinearVelocityTask2D, Core):
    """
    Implements the TrackXYVelocity task. The robot has to reach a target linear velocity.
    """

    def __init__(self, task_param: dict, reward_param: dict, num_envs: int, device: str):
        """
        Initializes the task.

        Args:
            task_param (dict): The parameters of the task.
            reward_param (dict): The parameters of the reward.
            num_envs (int): The number of environments.
            device (str): The device to run the task on.
        """
        Core.__init__(self, num_envs, device)
        # Task and reward parameters
        self._task_parameters = TrackLinearVelocityParameters(**task_param)
        self._reward_parameters = TrackLinearVelocityReward(**reward_param)

        # Define the specific observation space dimensions for this task
        self._dim_task_data = 3
        self.define_observation_space(self._dim_task_data)

        # Curriculum
        self._target_linear_velocity_sampler = CurriculumSampler(
            self._task_parameters.target_linear_velocity_curriculum,
        )
        self._spawn_linear_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_linear_velocity_curriculum,
        )
        self._spawn_angular_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_angular_velocity_curriculum,
        )

        # Buffers
        self._goal_reached = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)
        self._target_velocities = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self._task_label = self._task_label * 2

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

        self._velocity_error = self._target_velocities - current_state["linear_velocity"]
        self._position_error = current_state["position"]
        self._task_data[:, :3] = self._velocity_error
        return self.update_observation_tensor(current_state)

    def get_goals(
        self,
        env_ids: torch.Tensor,
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
        r = self._target_linear_velocity_sampler.sample(num_goals, step=step, device=self._device)
        theta = torch.rand((num_goals,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_goals,), device=self._device) * math.pi

        self._target_velocities[env_ids, 0] = r * torch.cos(theta) * torch.sin(phi)
        self._target_velocities[env_ids, 1] = r * torch.sin(theta) * torch.sin(phi)
        self._target_velocities[env_ids, 2] = r * torch.cos(phi)

        p = torch.zeros((num_goals, 3), device=self._device, dtype=torch.float32)
        q = torch.zeros((num_goals, 4), device=self._device, dtype=torch.float32)
        # This does not matter
        return p, q

    def get_initial_conditions(
        self, env_ids: torch.Tensor, step: int = 0
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
        initial_position = torch.zeros((num_resets, 3), device=self._device, dtype=torch.float32)
        # Randomizes the heading of the platform
        initial_orientation = torch.zeros((num_resets, 4), device=self._device, dtype=torch.float32)
        uvw = torch.rand((num_resets, 3), device=self._device)
        initial_orientation[:, 0] = torch.sqrt(uvw[:, 0]) * torch.cos(uvw[:, 2] * 2 * math.pi)
        initial_orientation[:, 1] = torch.sqrt(1 - uvw[:, 0]) * torch.sin(uvw[:, 1] * 2 * math.pi)
        initial_orientation[:, 2] = torch.sqrt(1 - uvw[:, 0]) * torch.cos(uvw[:, 1] * 2 * math.pi)
        initial_orientation[:, 3] = torch.sqrt(uvw[:, 0]) * torch.sin(uvw[:, 2] * 2 * math.pi)
        # Randomizes the linear velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)
        linear_velocity = self._spawn_linear_velocity_sampler.sample(num_resets, step, device=self._device)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        initial_velocity[:, 0] = linear_velocity * torch.cos(theta) * torch.sin(phi)
        initial_velocity[:, 1] = linear_velocity * torch.sin(theta) * torch.sin(phi)
        initial_velocity[:, 2] = linear_velocity * torch.cos(phi)
        # Randomizes the angular velocity of the platform
        angular_velocity = self._spawn_angular_velocity_sampler.sample(num_resets, step, device=self._device)
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
            dict: The dictionary containing the spawn data.
        """

        dict = {}

        num_resets = self._num_envs
        # Randomizes the linear velocity of the platform
        linear_velocities = self._spawn_linear_velocity_sampler.sample(num_resets, step, device=self._device)

        # Randomizes the angular velocity of the platform
        angular_velocities = self._spawn_angular_velocity_sampler.sample(num_resets, step, device=self._device)

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
            dict: The dictionary containing the target data.
        """

        dict = {}

        num_resets = self._num_envs
        # Randomizes the target linear velocity of the platform
        target_velocities = torch.zeros((num_resets, 3), device=self._device, dtype=torch.float32)
        r = self._target_linear_velocity_sampler.sample(num_resets, step=step, device=self._device)

        r = r.cpu().numpy()

        fig, ax = plt.subplots(dpi=100, figsize=(8, 8), sharey=True)
        ax.hist(r, bins=32)
        ax.set_title("Target normed linear velocity")
        ax.set_xlim(
            self._target_linear_velocity_sampler.get_min_bound(),
            self._target_linear_velocity_sampler.get_max_bound(),
        )
        ax.set_xlabel("vel (m/s)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/target_velocities"] = wandb.Image(data)
        return dict
