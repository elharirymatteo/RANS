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
    quat_to_mat,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_rewards import (
    GoToPoseReward,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_parameters import (
    GoToPoseParameters,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_go_to_pose import (
    GoToPoseTask as GoToPoseTask2D,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.curriculum_helpers import (
    CurriculumSampler,
)

from omniisaacgymenvs.utils.arrow3D import VisualArrow3D
from omni.isaac.core.prims import XFormPrimView
from pxr import Usd

from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import wandb
import torch
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPoseTask(GoToPoseTask2D, Core):
    """
    Implements the GoToPose task. The robot has to reach a target position and heading.
    """

    def __init__(
        self,
        task_param: dict,
        reward_param: dict,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Initializes the GoToPose task.

        Args:
            task_param (dict): The parameters of the task.
            reward_param (dict): The reward parameters of the task.
            num_envs (int): The number of environments.
            device (str): The device to run the task on.
        """

        Core.__init__(self, num_envs, device)
        # Task and reward parameters
        self._task_parameters = GoToPoseParameters(**task_param)
        self._reward_parameters = GoToPoseReward(**reward_param)
        # Curriculum samplers
        self._spawn_position_sampler = CurriculumSampler(
            self._task_parameters.spawn_position_curriculum
        )
        self._spawn_heading_sampler = CurriculumSampler(
            self._task_parameters.spawn_heading_curriculum
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
        self._target_headings = torch.zeros(
            (self._num_envs, 3, 3), device=self._device, dtype=torch.float32
        )
        self._target_quat = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
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

        # position distance
        self._position_error = self._target_positions - current_state["position"]
        # heading distance
        self._heading_error = torch.bmm(
            torch.transpose(current_state["orientation"], -2, -1), self._target_headings
        )
        # Encode task data
        self._task_data[:, :3] = self._position_error
        self._task_data[:, 3:] = self._heading_error[:, :2, :].reshape(
            self._num_envs, 6
        )
        return self.update_observation_tensor(current_state)

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        step: int = 0,
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int. optional): The current training step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot.
        """

        # position error
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        trace = (
            self._heading_error[:, 0, 0]
            + self._heading_error[:, 1, 1]
            + self._heading_error[:, 2, 2]
        )
        self.heading_dist = torch.arccos((trace - 1) / 2)
        # boundary penalty
        self.boundary_dist = torch.abs(
            self._task_parameters.kill_dist - self.position_dist
        )
        self.boundary_penalty = self._task_parameters.boundary_penalty.compute_penalty(
            self.boundary_dist, step
        )

        # Checks if the goal is reached
        position_goal_is_reached = (
            self.position_dist < self._task_parameters.position_tolerance
        ).int()
        heading_goal_is_reached = (
            self.heading_dist < self._task_parameters.orientation_tolerance
        ).int()
        goal_is_reached = position_goal_is_reached * heading_goal_is_reached
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        # rewards
        (
            self.position_reward,
            self.heading_reward,
        ) = self._reward_parameters.compute_reward(
            current_state, actions, self.position_dist, self.heading_dist
        )
        return self.position_reward + self.heading_reward - self.boundary_penalty

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a random goal for the task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            target_positions (torch.Tensor): The target positions of the environments.
            target_orientations (torch.Tensor): The target orientations of the environments.
            step (int, optional): The current training step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations.
        """

        num_goals = len(env_ids)
        # Randomize position
        self._target_positions[env_ids] = (
            torch.rand((num_goals, 3), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        target_positions[env_ids, :3] += self._target_positions[env_ids]
        # Randomize heading
        uvw = torch.rand((num_goals, 3), device=self._device)
        quat = torch.zeros((num_goals, 4), device=self._device)
        quat[:, 0] = torch.sqrt(uvw[:, 0]) * torch.cos(uvw[:, 2] * 2 * math.pi)
        quat[:, 1] = torch.sqrt(1 - uvw[:, 0]) * torch.sin(uvw[:, 1] * 2 * math.pi)
        quat[:, 2] = torch.sqrt(1 - uvw[:, 0]) * torch.cos(uvw[:, 1] * 2 * math.pi)
        quat[:, 3] = torch.sqrt(uvw[:, 0]) * torch.sin(uvw[:, 2] * 2 * math.pi)
        # cast quaternions to rotation matrix
        self._target_quat[env_ids] = quat
        self._target_headings = quat_to_mat(quat)
        return target_positions, quat

    def get_initial_conditions(
        self,
        env_ids: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates spawning positions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current training step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial positions, orientations, and velocities.
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
        initial_position[:, 0] = (
            r * torch.cos(theta) * torch.sin(phi) + self._target_positions[0]
        )
        initial_position[:, 1] = (
            r * torch.sin(theta) * torch.sin(phi) + self._target_positions[1]
        )
        initial_position[:, 2] = r * torch.cos(phi) + self._target_positions[2]

        # Randomizes the orientation of the platform
        # We want to sample something that's not too far from the original orientation
        initial_orientation = torch.zeros(
            (num_resets, 4), device=self._device, dtype=torch.float32
        )
        d = self._spawn_position_sampler.sample(num_resets, step, device=self._device)

        uvw = torch.rand((num_resets, 3), device=self._device)
        uvw = uvw / torch.norm(uvw, dim=-1, keepdim=True)

        initial_orientation[:, 0] = np.cos(d / 2)
        initial_orientation[:, 1] = uvw[:, 0] * np.sin(d / 2)
        initial_orientation[:, 2] = uvw[:, 1] * np.sin(d / 2)
        initial_orientation[:, 3] = uvw[:, 2] * np.sin(d / 2)

        initial_orientation = initial_orientation * self._target_quat[env_ids]

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
        An arrow is generated to represent the 2D pose to be reached by the agent."""

        color = torch.tensor([1, 0, 0])
        body_radius = 0.025
        body_length = 1.5
        head_radius = 0.075
        head_length = 0.5
        VisualArrow3D(
            prim_path=path + "/arrow",
            translation=position,
            name="target_0",
            body_radius=body_radius,
            body_length=body_length,
            head_radius=head_radius,
            head_length=head_length,
            color=color,
        )

    def add_visual_marker_to_scene(self, scene):
        """
        Adds the visual marker to the scene."""

        arrows = XFormPrimView(prim_paths_expr="/World/envs/.*/arrow")
        scene.add(arrows)
        return scene, arrows

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
        xyz_pos = torch.zeros((num_resets, 2), device=self._device, dtype=torch.float32)
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        xyz_pos[:, 0] = r * torch.cos(theta) * torch.sin(phi)
        xyz_pos[:, 1] = r * torch.sin(theta) * torch.sin(phi)
        xyz_pos[:, 2] = r * torch.cos(phi)
        # Randomizes the heading of the platform
        heading = self._spawn_heading_sampler.sample(
            num_resets, step, device=self._device
        )
        # Randomizes the linear velocity of the platform
        velocities = torch.zeros(
            (num_resets, 6), device=self._device, dtype=torch.float32
        )
        linear_velocity = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        velocities[:, 0] = linear_velocity * torch.cos(theta) * torch.sin(phi)
        velocities[:, 1] = linear_velocity * torch.sin(theta) * torch.sin(phi)
        velocities[:, 2] = linear_velocity * torch.cos(phi)
        # Randomizes the angular velocity of the platform
        angular_velocity = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        velocities[:, 3] = angular_velocity * torch.cos(theta) * torch.sin(phi)
        velocities[:, 4] = angular_velocity * torch.sin(theta) * torch.sin(phi)
        velocities[:, 5] = angular_velocity * torch.cos(phi)

        xy_pos = xy_pos.cpu().numpy()
        heading = np.expand_dims(heading.cpu().numpy(), axis=-1)
        velocities = velocities.cpu().numpy()

        fig.tight_layout()
        fig, ax = plt.subplots(1, 3, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(xyz_pos[:, 0], bins=32)
        ax[0].set_title("Initial x position")
        ax[0].set_xlim(self._task_parameters.kill_dist, self._task_parameters.kill_dist)
        ax[0].set_xlabel("pos (m)")
        ax[0].set_ylabel("count")
        ax[1].hist(xyz_pos[:, 1], bins=32)
        ax[1].set_title("Initial y position")
        ax[1].set_xlim(self._task_parameters.kill_dist, self._task_parameters.kill_dist)
        ax[1].set_xlabel("pos (m)")
        ax[2].hist(xyz_pos[:, 2], bins=32)
        ax[2].set_title("Initial z position")
        ax[2].set_xlim(self._task_parameters.kill_dist, self._task_parameters.kill_dist)
        ax[2].set_xlabel("pos (m)")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_position"] = wandb.Image(data)

        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(heading[:, 0], bins=32)
        ax.set_title("Initial heading")
        ax.set_xlim(-math.pi, math.pi)
        ax.set_xlabel("theta (rad)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_heading"] = wandb.Image(data)

        fig, ax = plt.subplots(1, 3, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(velocities[:, 0], bins=32)
        ax[0].set_title("Initial x linear velocity")
        ax[0].set_xlim(-0.5, 0.5)
        ax[0].set_xlabel("vel (m/s)")
        ax[0].set_ylabel("count")
        ax[1].hist(velocities[:, 1], bins=32)
        ax[1].set_title("Initial y linear velocity")
        ax[1].set_xlim(-0.5, 0.5)
        ax[1].set_xlabel("vel (m/s)")
        ax[2].hist(velocities[:, 2], bins=32)
        ax[2].set_title("Initial z linear velocity")
        ax[2].set_xlim(-0.5, 0.5)
        ax[2].set_xlabel("vel (rad/s)")
        fig.tight_layout()
        fig.canvas.draw()

        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_linear_velocities"] = wandb.Image(data)

        fig, ax = plt.subplots(1, 3, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(velocities[:, 0], bins=32)
        ax[0].set_title("Initial x angular velocity")
        ax[0].set_xlim(-0.5, 0.5)
        ax[0].set_xlabel("vel (m/s)")
        ax[0].set_ylabel("count")
        ax[1].hist(velocities[:, 1], bins=32)
        ax[1].set_title("Initial y angular velocity")
        ax[1].set_xlim(-0.5, 0.5)
        ax[1].set_xlabel("vel (m/s)")
        ax[2].hist(velocities[:, 2], bins=32)
        ax[2].set_title("Initial z angular velocity")
        ax[2].set_xlim(-0.5, 0.5)
        ax[2].set_xlabel("vel (rad/s)")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_angular_velocities"] = wandb.Image(data)
        return dict
