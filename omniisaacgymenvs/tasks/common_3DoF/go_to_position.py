__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_3DoF.core import (
    Core,
)
from omniisaacgymenvs.tasks.common_3DoF.task_rewards import (
    GoToPositionReward,
)
from omniisaacgymenvs.tasks.common_3DoF.task_parameters import (
    GoToPositionParameters,
)
from omniisaacgymenvs.tasks.common_3DoF.curriculum_helpers import (
    CurriculumSampler,
)

from omniisaacgymenvs.utils.pin import VisualPin
from omni.isaac.core.prims import XFormPrimView
from pxr import Usd

from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import wandb
import torch
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPositionTask(Core):
    """
    Implements the GoToXY task. The robot has to reach a target position.
    """

    def __init__(
        self,
        task_param: dict,
        reward_param: dict,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Initializes the GoToXY task.

        Args:
            task_param (dict): Dictionary containing the task parameters.
            reward_param (dict): Dictionary containing the reward parameters.
            num_envs (int): Number of environments.
            device (str): Device to run the task on.
        """

        super(GoToPositionTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = GoToPositionParameters(**task_param)
        self._reward_parameters = GoToPositionReward(**reward_param)
        
        # Define the specific observation space dimensions for this task
        self._dim_task_data = 2
        self.define_observation_space(self._dim_task_data)

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
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 0

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task.

        Args:
            stats (dict): Dictionary containing the statistics.

        Returns:
            dict: Dictionary containing the statistics.
        """

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )

        if not "position_reward" in stats.keys():
            stats["position_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "boundary_dist" in stats.keys():
            stats["boundary_dist"] = torch_zeros()
        self.log_with_wandb = []
        self.log_with_wandb += self._task_parameters.boundary_penalty.get_stats_name()
        for name in self._task_parameters.boundary_penalty.get_stats_name():
            if not name in stats.keys():
                stats[name] = torch_zeros()
        return stats

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            current_state (dict): The current state of the robot.

        Returns:
            torch.Tensor: The observation tensor.
        """

        self._position_error = self._target_positions - current_state["position"]
        self._task_data[:, :2] = self._position_error
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
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot.
        """

        # position error
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        # boundary penalty
        self.boundary_dist = torch.abs(
            self._task_parameters.kill_dist - self.position_dist
        )
        self.boundary_penalty = self._task_parameters.boundary_penalty.compute_penalty(
            self.boundary_dist, step
        )

        # Checks if the goal is reached
        goal_is_reached = (
            self.position_dist < self._task_parameters.position_tolerance
        ).int()
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        # Rewards
        self.position_reward = self._reward_parameters.compute_reward(
            current_state, actions, self.position_dist
        )

        return self.position_reward - self.boundary_penalty

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

        stats["position_reward"] += self.position_reward
        stats["position_error"] += self.position_dist
        stats["boundary_dist"] += self.boundary_dist
        stats = self._task_parameters.boundary_penalty.update_statistics(stats)
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
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a random goal for the task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations.
        """

        num_goals = len(env_ids)
        self._target_positions[env_ids] = (
            torch.rand((num_goals, 2), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        p = torch.zeros((num_goals, 3), dtype=torch.float32, device=self._device)
        p[:, :2] += self._target_positions[env_ids]
        p[:, 2] = 2
        q = torch.zeros((num_goals, 4), dtype=torch.float32, device=self._device)
        q[:, 0] = 1

        return p, q

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
        self.reset(env_ids)
        # Ranomizes the starting position of the platform
        initial_position = torch.zeros(
            (num_resets, 3), device=self._device, dtype=torch.float32
        )
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_position[:, 0] = r * torch.cos(theta)
        initial_position[:, 1] = r * torch.sin(theta)
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

        return (
            initial_position,
            initial_orientation,
            initial_velocity,
        )

    def generate_target(self, path: str, position: torch.Tensor) -> None:
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.
        A pin is generated to represent the 2D position to be reached by the agent.

        Args:
            path (str): The path where the pin is to be generated.
            position (torch.Tensor): The position of the pin.
        """

        color = torch.tensor([1, 0, 0])
        ball_radius = 0.2
        poll_radius = 0.025
        poll_length = 2
        VisualPin(
            prim_path=path + "/pin",
            translation=position,
            name="target_0",
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            color=color,
        )

    def add_visual_marker_to_scene(
        self, scene: Usd.Stage
    ) -> Tuple[Usd.Stage, XFormPrimView]:
        """
        Adds the visual marker to the scene.

        Args:
            scene (Usd.Stage): The scene to add the visual marker to.

        Returns:
            Tuple[Usd.Stage, XFormPrimView]: The scene and the visual marker.
        """

        pins = XFormPrimView(prim_paths_expr="/World/envs/.*/pin")
        scene.add(pins)
        return scene, pins

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
        ax.set_title("Spawn position")
        ax.set_xlim(
            self._spawn_position_sampler.get_min_bound(),
            self._spawn_position_sampler.get_max_bound(),
        )
        ax.set_xlabel("spawn distance (m)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/spawn_position"] = wandb.Image(data)

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

        dict["curriculum/initial_linear_velocities"] = wandb.Image(data)
        return dict

    def log_target_data(self, step: int) -> dict:
        """
        Logs the target data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The target data.
        """

        return {}

    def get_logs(self, step: int) -> dict:
        """
        Logs the task data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The task data.
        """

        dict = self._task_parameters.boundary_penalty.get_logs()
        if step % 50 == 0:
            dict = {**dict, **self.log_spawn_data(step)}
            dict = {**dict, **self.log_target_data(step)}
        return dict
