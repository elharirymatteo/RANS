__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


from omniisaacgymenvs.tasks.MFP.MFP2D_core import (
    Core,
)
from omniisaacgymenvs.tasks.MFP.MFP2D_task_rewards import (
    GoThroughXYSequenceReward,
)
from omniisaacgymenvs.tasks.MFP.MFP2D_task_parameters import (
    GoThroughXYSequenceParameters,
)
from omniisaacgymenvs.tasks.MFP.curriculum_helpers import (
    CurriculumSampler,
)

from omniisaacgymenvs.utils.pin import VisualPin
from omni.isaac.core.prims import XFormPrimView
from pxr import Usd

from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import colorsys
import wandb
import torch
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoThroughXYSequenceTask(Core):
    """
    Implements the GoThroughXYSequence task. The robot has to reach a sequence of points in the 2D plane
    at a given velocity, it must do so while looking at the target. Unlike the GoThroughXY task, the robot
    has to reach a sequence of points in the 2D plane.
    """

    def __init__(
        self,
        task_param: dict,
        reward_param: dict,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Initializes the GoThroughXYSequence task.

        Args:
            task_param (dict): The parameters of the task.
            reward_param (dict): The reward parameters of the task.
            num_envs (int): The number of environments.
            device (str): The device to run the task on.
        """

        super(GoThroughXYSequenceTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = GoThroughXYSequenceParameters(**task_param)
        self._reward_parameters = GoThroughXYSequenceReward(**reward_param)
        # Curriculum samplers
        self._spawn_position_sampler = CurriculumSampler(
            self._task_parameters.spawn_position_curriculum
        )
        self._target_linear_velocity_sampler = CurriculumSampler(
            self._task_parameters.target_linear_velocity_curriculum,
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
        self._all = torch.arange(self._num_envs, device=self._device)
        self._trajectory_completed = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, self._task_parameters.num_points, 2),
            device=self._device,
            dtype=torch.float32,
        )
        self._target_index = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.long
        )
        self._target_headings = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._target_velocities = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._delta_headings = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._previous_position_dist = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 5

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task.

        Args:
            stats (dict): The dictionary to store the statistics.

        Returns:used
            dict: The dictionary containing the statistics.
        """

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )
        if not "progress_reward" in stats.keys():
            stats["progress_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "heading_reward" in stats.keys():
            stats["heading_reward"] = torch_zeros()
        if not "linear_velocity_reward" in stats.keys():
            stats["linear_velocity_reward"] = torch_zeros()
        if not "linear_velocity_error" in stats.keys():
            stats["linear_velocity_error"] = torch_zeros()
        if not "heading_error" in stats.keys():
            stats["heading_error"] = torch_zeros()
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

        # position distance
        self._position_error = (
            self._target_positions[self._all, self._target_index]
            - current_state["position"]
        ).squeeze()
        # linear velocity error (normed velocity)
        self.linear_velocity_err = self._target_velocities - torch.norm(
            current_state["linear_velocity"], dim=-1
        )
        # heading distance
        heading = torch.arctan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        # Compute target heading as the angle required to be looking at the target
        self._target_headings = torch.arctan2(
            self._position_error[:, 1], self._position_error[:, 0]
        )
        self._heading_error = torch.arctan2(
            torch.sin(self._target_headings - heading),
            torch.cos(self._target_headings - heading),
        )

        # Encode task data
        self._task_data[:, :2] = self._position_error
        self._task_data[:, 2] = torch.cos(self._heading_error)
        self._task_data[:, 3] = torch.sin(self._heading_error)
        self._task_data[:, 4] = self.linear_velocity_err
        # position of the other points in the sequence
        for i in range(self._task_parameters.num_points - 1):
            overflowing = self._target_index + i + 1 >= self._task_parameters.num_points
            indices = self._target_index + (i + 1) * (1 - overflowing.int())
            self._task_data[:, 5 + 2 * i : 5 + 2 * (i + 1)] = (
                self._target_positions[self._all, indices] - current_state["position"]
            ) * (1 - overflowing.int()).view(-1, 1)

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
        # Compute progress and normalize by the target velocity
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        self.linear_velocity_dist = torch.abs(self.linear_velocity_err)
        position_progress = (
            self._previous_position_dist - self.position_dist
        ) / torch.abs(self._target_velocities)
        was_killed = (self._previous_position_dist == 0).float()
        position_progress = position_progress * (1 - was_killed)
        # Heading
        self.heading_dist = torch.abs(self._heading_error)
        # boundary penalty
        self.boundary_dist = torch.abs(
            self._task_parameters.kill_dist - self.position_dist
        )
        self.boundary_penalty = self._task_parameters.boundary_penalty.compute_penalty(
            self.boundary_dist, step
        )

        # Checks if the goal is reached
        goal_reached = (
            self.position_dist < self._task_parameters.position_tolerance
        ).int()
        reached_ids = goal_reached.nonzero(as_tuple=False).squeeze(-1)
        # if the goal is reached, the target index is updated
        self._target_index = self._target_index + goal_reached
        self._trajectory_completed = (
            self._target_index >= self._task_parameters.num_points
        ).int()

        # rewards
        (
            self.progress_reward,
            self.heading_reward,
            self.linear_velocity_reward,
        ) = self._reward_parameters.compute_reward(
            current_state,
            actions,
            position_progress,
            self.heading_dist,
            self.linear_velocity_dist,
        )
        self._previous_position_dist = self.position_dist.clone()
        # If goal is reached make next progress null
        self._previous_position_dist[reached_ids] = 0
        return (
            self.progress_reward
            + self.heading_reward
            - self.boundary_penalty
            - self._reward_parameters.time_penalty
            + self._trajectory_completed * self._reward_parameters.terminal_reward
        )

    def update_kills(self) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not.
        """

        die = torch.zeros_like(self._trajectory_completed, dtype=torch.long)
        ones = torch.ones_like(self._trajectory_completed, dtype=torch.long)
        die = torch.where(
            self.position_dist > self._task_parameters.kill_dist, ones, die
        )
        die = torch.where(self._trajectory_completed > 0, ones, die)
        return die

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict):The new stastistics to be logged.

        Returns:
            dict: The statistics of the training
        """

        stats["progress_reward"] += self.progress_reward
        stats["heading_reward"] += self.heading_reward
        stats["linear_velocity_reward"] += self.linear_velocity_reward
        stats["position_error"] += self.position_dist
        stats["heading_error"] += self.heading_dist
        stats["linear_velocity_error"] += self.linear_velocity_dist
        stats["boundary_dist"] += self.boundary_dist
        stats = self._task_parameters.boundary_penalty.update_statistics(stats)
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
        """
        self._trajectory_completed[env_ids] = 0
        self._target_index[env_ids] = 0
        self._previous_position_dist[env_ids] = 0

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
        # Randomize position
        for i in range(self._task_parameters.num_points):
            if i == 0:
                self._target_positions[env_ids, i] = (
                    torch.rand((num_goals, 2), device=self._device)
                    * self._task_parameters.goal_random_position
                    * 2
                    - self._task_parameters.goal_random_position
                )
            else:
                r = self._spawn_position_sampler.sample(
                    num_goals, step, device=self._device
                )
                theta = torch.rand((num_goals,), device=self._device) * 2 * math.pi
                point = torch.zeros((num_goals, 2), device=self._device)
                point[:, 0] = r * torch.cos(theta)
                point[:, 1] = r * torch.sin(theta)
                self._target_positions[env_ids, i] = (
                    self._target_positions[env_ids, i - 1] + point
                )

        # Randomize heading
        self._delta_headings[env_ids] = self._spawn_heading_sampler.sample(
            num_goals, step, device=self._device
        )
        # Randomizes the target linear velocity
        r = self._target_linear_velocity_sampler.sample(
            num_goals, step=step, device=self._device
        )
        self._target_velocities[env_ids] = r

        # Creates tensors to save position and orientation
        p = torch.zeros(
            (num_goals, self._task_parameters.num_points, 3), device=self._device
        )
        q = torch.zeros(
            (num_goals, self._task_parameters.num_points, 4),
            device=self._device,
            dtype=torch.float32,
        )
        q[:, :, 0] = 1
        p[:, :, :2] = self._target_positions[env_ids]

        return p, q

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
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_position[:, 0] = (
            r * torch.cos(theta) + self._target_positions[env_ids, 0, 0]
        )
        initial_position[:, 1] = (
            r * torch.sin(theta) + self._target_positions[env_ids, 0, 1]
        )
        # Randomizes the heading of the platform
        initial_orientation = torch.zeros(
            (num_resets, 4), device=self._device, dtype=torch.float32
        )
        target_position_local = (
            self._target_positions[env_ids, 0, :2] - initial_position[:, :2]
        )
        target_heading = torch.arctan2(
            target_position_local[:, 1], target_position_local[:, 0]
        )
        theta = target_heading + self._delta_headings[env_ids]
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
        An arrow is generated to represent the 3DoF pose to be reached by the agent.

        Args:
            path (str): The path where the pin is to be generated.
            position (torch.Tensor): The position of the arrow.
        """

        for i in range(self._task_parameters.num_points):
            color = torch.tensor(
                colorsys.hsv_to_rgb(i / self._task_parameters.num_points, 1, 1)
            )
            ball_radius = 0.2
            poll_radius = 0.025
            poll_length = 2
            VisualPin(
                prim_path=path + "/pin_" + str(i),
                translation=position,
                name="target_" + str(i),
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

        pins = XFormPrimView(prim_paths_expr="/World/envs/.*/pin_[0-5]")
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
        # Randomizes the heading of the platform
        heading = self._spawn_heading_sampler.sample(
            num_resets, step, device=self._device
        )
        # Randomizes the linear velocity of the platform
        linear_velocities = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        # Randomizes the angular velocity of the platform
        angular_velocities = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )

        r = r.cpu().numpy()
        heading = heading.cpu().numpy()
        linear_velocities = linear_velocities.cpu().numpy()
        angular_velocities = angular_velocities.cpu().numpy()

        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(r, bins=32)
        ax.set_title("Initial position")
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

        dict["curriculum/initial_position"] = wandb.Image(data)

        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(heading, bins=32)
        ax.set_title("Initial heading")
        ax.set_xlim(
            self._spawn_heading_sampler.get_min_bound(),
            self._spawn_heading_sampler.get_max_bound(),
        )
        ax.set_xlabel("angular distance (rad)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_heading"] = wandb.Image(data)

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
            num_resets, step=step, device=self._device
        )

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
