__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_core import (
    Core,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_task_rewards import (
    TrackXYOVelocityReward,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_task_parameters import (
    TrackXYOVelocityParameters,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.curriculum_helpers import (
    CurriculumSampler,
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


class TrackXYOVelocityTask(Core):
    """
    Implements the GoToPose task. The robot has to reach a target position and heading.
    """

    def __init__(self, task_param, reward_param, num_envs, device):
        super(TrackXYOVelocityTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = TrackXYOVelocityParameters(**task_param)
        self._reward_parameters = TrackXYOVelocityReward(**reward_param)

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
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self._target_angular_velocities = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 3

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task.

        Args:
            stats (dict): The dictionary to store the statistics.

        Returns:
            dict: The dictionary containing the statistics.
        """

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )

        if not "linear_velocity_reward" in stats.keys():
            stats["linear_velocity_reward"] = torch_zeros()
        if not "linear_velocity_error" in stats.keys():
            stats["linear_velocity_error"] = torch_zeros()
        if not "angular_velocity_reward" in stats.keys():
            stats["angular_velocity_reward"] = torch_zeros()
        if not "angular_velocity_error" in stats.keys():
            stats["angular_velocity_error"] = torch_zeros()
        return stats

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
        self._task_data[:, :2] = self._linear_velocity_error
        self._task_data[:, 2] = self._angular_velocity_error
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

    def update_kills(self) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not.
        """

        die = torch.zeros_like(self._goal_reached, dtype=torch.long)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        die = torch.where(
            self.position_dist > self._task_parameters.kill_dist, ones, die
        )
        die = torch.where(
            self._goal_reached > self._task_parameters.kill_after_n_steps_in_tolerance,
            ones,
            die,
        )
        return die

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict):The new stastistics to be logged.

        Returns:
            dict: The statistics of the training
        """

        stats["linear_velocity_reward"] += self.linear_velocity_reward
        stats["linear_velocity_error"] += self.linear_velocity_dist
        stats["angular_velocity_reward"] += self.angular_velocity_reward
        stats["angular_velocity_error"] += self.angular_velocity_dist
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
        """

        self._goal_reached[env_ids] = 0

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
            num_goals, step=0, device=self._device
        )
        theta = torch.rand((num_goals,), device=self._device) * 2 * math.pi
        self._target_linear_velocities[env_ids, 0] = r * torch.cos(theta)
        self._target_linear_velocities[env_ids, 1] = r * torch.sin(theta)
        # Randomizes the target angular velocity
        omega = self._target_angular_velocity_sampler.sample(
            num_goals, step=0, device=self._device
        )
        self._target_angular_velocities[env_ids] = omega

        if math.fmod(step, 50) == 0:
            self.log_target_data(int(step))

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
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_orientation[:, 0] = torch.cos(theta * 0.5)
        initial_orientation[:, 3] = torch.sin(theta * 0.5)
        # Randomizes the linear velocity of the platform
        initial_velocity = torch.zeros(
            (num_resets, 6), device=self._device, dtype=torch.float32
        )
        linear_velocity = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[:, 0] = linear_velocity * torch.cos(theta)
        initial_velocity[:, 1] = linear_velocity * torch.sin(theta)
        # Randomizes the angular velocity of the platform
        angular_velocity = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        initial_velocity[:, 5] = angular_velocity

        if math.fmod(step, 50) == 0:
            self.log_spawn_data(int(step))

        return (
            initial_position,
            initial_orientation,
            initial_velocity,
        )

    def generate_target(self, path: str, position: torch.Tensor) -> None:
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.

        Args:
            path (str): The path where the pin is to be generated.
            position (torch.Tensor): The position of the target.
        """

        pass

    def add_visual_marker_to_scene(self, scene: Usd.Stage) -> Tuple[Usd.Stage, None]:
        """
        Adds the visual marker to the scene.

        Args:
            scene (Usd.Stage): The scene to add the visual marker to.

        Returns:
            Tuple[Usd.Stage, None]: The scene and the visual marker.
        """

        return scene, None

    def log_spawn_data(self, step: int) -> None:
        """
        Logs the spawn data to wandb.

        Args:
            step (int): The current step.
        """

        num_resets = self._num_envs
        # Randomizes the linear velocity of the platform
        xyz_velocity = torch.zeros(
            (num_resets, 3), device=self._device, dtype=torch.float32
        )
        linear_velocity = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        xyz_velocity[:, 0] = linear_velocity * torch.cos(theta)
        xyz_velocity[:, 1] = linear_velocity * torch.sin(theta)
        # Randomizes the angular velocity of the platform
        angular_velocity = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        xyz_velocity[:, 2] = angular_velocity

        xy_pos = xy_pos.cpu().numpy()
        heading = np.expand_dims(heading.cpu().numpy(), axis=-1)
        xyz_velocity = xyz_velocity.cpu().numpy()

        fig, ax = plt.subplots(1, 3, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(xyz_velocity[:, 0], bins=32)
        ax[0].set_title("Initial x linear velocity")
        ax[0].set_xlim(-0.5, 0.5)
        ax[0].set_xlabel("vel (m/s)")
        ax[0].set_ylabel("count")
        ax[1].hist(xyz_velocity[:, 1], bins=32)
        ax[1].set_title("Initial y linear velocity")
        ax[1].set_xlim(-0.5, 0.5)
        ax[1].set_xlabel("vel (m/s)")
        ax[2].hist(xyz_velocity[:, 2], bins=32)
        ax[2].set_title("Initial z angular velocity")
        ax[2].set_xlim(-0.5, 0.5)
        ax[2].set_xlabel("vel (rad/s)")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())

        wandb.log(
            {
                "curriculum/initial_velocities": wandb.Image(data),
            }
        )

    def log_target_data(self, step: int) -> None:
        """
        Logs the target data to wandb.

        Args:
            step (int): The current step.
        """

        num_resets = self._num_envs
        # Randomizes the target linear velocity of the platform
        xyz_velocity = torch.zeros(
            (num_resets, 3), device=self._device, dtype=torch.float32
        )
        linear_velocity = self._target_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        xyz_velocity[:, :2] = linear_velocity
        # Randomizes the angular velocity of the platform
        angular_velocity = self._target_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        xyz_velocity[:, 2] = angular_velocity
        xyz_velocity = xyz_velocity.cpu().numpy()

        fig, ax = plt.subplots(1, 3, dpi=100, figsize=(8, 8), sharey=True)
        ax[0].hist(xyz_velocity[:, 0], bins=32)
        ax[0].set_title("Target x linear velocity")
        ax[0].set_xlim(-0.5, 0.5)
        ax[0].set_xlabel("vel (m/s)")
        ax[0].set_ylabel("count")
        ax[1].hist(xyz_velocity[:, 1], bins=32)
        ax[1].set_title("Target y linear velocity")
        ax[1].set_xlim(-0.5, 0.5)
        ax[1].set_xlabel("vel (m/s)")
        ax[2].hist(xyz_velocity[:, 2], bins=32)
        ax[2].set_title("Target z angular velocity")
        ax[2].set_xlim(-0.5, 0.5)
        ax[2].set_xlabel("vel (rad/s)")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())

        wandb.log(
            {
                "curriculum/target_velocities": wandb.Image(data),
            }
        )
