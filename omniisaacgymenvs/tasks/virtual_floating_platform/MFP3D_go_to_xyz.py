__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_core import (
    Core,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_rewards import (
    GoToXYZReward,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_parameters import (
    GoToXYZParameters,
)

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_go_to_xy import (
    GoToXYTask as GoToXYTask2D,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.curriculum_helpers import (
    CurriculumSampler,
)

from omniisaacgymenvs.utils.pin3D import VisualPin3D
from omni.isaac.core.prims import XFormPrimView
from pxr import Usd
from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import wandb
import torch
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToXYZTask(GoToXYTask2D, Core):
    """
    Implements the GoToXY task. The robot has to reach a target position."""

    def __init__(
        self,
        task_param: dict,
        reward_param: dict,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Initializes the GoToXYZ task.

        Args:
            task_param (dict): Dictionary containing the task parameters.
            reward_param (dict): Dictionary containing the reward parameters.
            num_envs (int): Number of environments.
            device (str): Device to run the task on.
        """

        Core.__init__(self, num_envs, device)
        # Task and reward parameters
        self._task_parameters = GoToXYZParameters(**task_param)
        self._reward_parameters = GoToXYZReward(**reward_param)
        # Curriculum samplers
        self._spawn_position_sampler = CurriculumSampler(
            self._task_parameters.spawn_position_curriculum
        )
        self._spawn_linear_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_linear_velocity_curriculum
        )
        self._spawn_angular_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_angular_velocity_curriculum
        )

        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 1

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

        self._position_error = self._target_positions - current_state["position"]
        self._task_data[:, :3] = self._position_error
        return self.update_observation_tensor(current_state)

    def get_goals(
        self,
        env_ids: torch.Tensor,
        targets_position: torch.Tensor,
        targets_orientation: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a random goal for the task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            target_positions (torch.Tensor): The target positions.
            target_orientations (torch.Tensor): The target orientations.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations.
        """

        num_goals = len(env_ids)
        self._target_positions[env_ids] = (
            torch.rand((num_goals, 3), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        targets_position[env_ids, :3] += self._target_positions[env_ids]
        return targets_position, targets_orientation

    def get_initial_conditions(
        self,
        env_ids: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates spawning positions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot.
        """

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self._goal_reached[env_ids] = 0
        # Randomizes the starting position of the platform
        initial_position = torch.zeros(
            (num_resets, 3), device=self._device, dtype=torch.float32
        )
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        initial_position[:, 0] = r * torch.cos(theta) * torch.sin(phi)
        initial_position[:, 1] = r * torch.sin(theta) * torch.sin(phi)
        initial_position[:, 1] = r * torch.cos(phi)
        # Randomizes the orientation of the platform
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
        return initial_position, initial_orientation, initial_velocity

    def generate_target(self, path, position):
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.
        A pin is generated to represent the 3D position to be reached by the agent."""

        color = torch.tensor([1, 0, 0])
        ball_radius = 0.05
        poll_radius = 0.025
        poll_length = 2
        VisualPin3D(
            prim_path=path + "/pin",
            translation=position,
            name="target_0",
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            color=color,
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
        # Resets the counter of steps for which the goal was reached
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        # Randomizes the linear velocity of the platform
        linear_velocities = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        # Randomizes the angular velocity of the platform
        angular_velocities = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )

        r = r.cpu().numpy()
        linear_velocities = linear_velocities.cpu().numpy()
        angular_velocities = angular_velocities.cpu().numpy()

        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(r, bins=32)
        ax.set_title("Spawn radius")
        ax.set_xlabel("spawn distance (m)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/spawn_position"] = wandb.Image(data)

        fig, ax = plt.subplots(1, 2, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(linear_velocities[:, 0], bins=32)
        ax[0].set_title("Initial normed linear velocity")
        ax[0].set_xlim(
            self._spawn_linear_velocity_sampler.get_min_bound(),
            self._spawn_linear_velocity_sampler.get_max_bound(),
        )
        ax[0].set_xlabel("vel (m/s)")
        ax[0].set_ylabel("count")
        ax[1].hist(angular_velocities[:, 1], bins=32)
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
