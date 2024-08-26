__author__ = "Luis Batista"
__copyright__ = (
    "Copyright 2024, DREAM Lab, Georgia Tech Europe"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Luis Batista"
__email__ = "luis.batista@gatech.edu"
__status__ = "development"


from omniisaacgymenvs.tasks.common_3DoF.core import Core
from omniisaacgymenvs.tasks.ASV.task_rewards import GoThroughPositionReward
from omniisaacgymenvs.tasks.common_3DoF.task_parameters import GoThroughPositionParameters
from omniisaacgymenvs.tasks.common_3DoF.curriculum_helpers import CurriculumSampler

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

class GoThroughPositionTask(Core):

    def __init__(self, task_param: dict, reward_param: dict, num_envs: int, device: str) -> None:
        """
        Initializes the GoToPose task.

        Args:
            task_param (dict): The parameters of the task.
            reward_param (dict): The reward parameters of the task.
            num_envs (int): The number of environments.
            device (str): The device to run the task on.
        """

        super(GoThroughPositionTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = GoThroughPositionParameters(**task_param)
        self._reward_parameters = GoThroughPositionReward(**reward_param)

        # Define the specific observation space dimensions for this task
        self.define_observation_space()

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
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
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
        if not "energy_sum" in stats.keys():
            stats["energy_sum"] = torch_zeros()
        self.log_with_wandb = []
        self.log_with_wandb += self._task_parameters.boundary_penalty.get_stats_name()
        for name in self._task_parameters.boundary_penalty.get_stats_name():
            if not name in stats.keys():
                stats[name] = torch_zeros()
        return stats


    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Returns the state observations.
        """

        self.update_observation_tensor(current_state)

        return self._obs_buffer
    
    def compute_reward(self, current_state: torch.Tensor, actions: torch.Tensor, step: int = 0) -> torch.Tensor:
        # TODO: Just making it run... understand and simplify the rewards.
        self.position_dist = self.target_distance
        self._heading_error = self.target_bearing_rad

        position_progress = (
            self._previous_position_dist - self.position_dist
        )

        was_killed = (self._previous_position_dist == 0).float()
        position_progress = position_progress * (1 - was_killed)

        # boundary penalty
        self.heading_dist = torch.abs(self._heading_error)
        self.boundary_dist = torch.abs(self._task_parameters.kill_dist - self.position_dist)
        self.boundary_penalty = self._task_parameters.boundary_penalty.compute_penalty(self.boundary_dist, step)
        self.linear_velocity_dist = torch.abs(self.linear_velocity_error)
        self.energy_sum = actions.pow(2).sum(dim=-1).sqrt().mean()

        # Checks if the goal is reached
        self._goal_reached = (self.position_dist < self._task_parameters.position_tolerance).int()

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
        reward = (
            self.progress_reward
            + self.heading_reward
            + self.linear_velocity_reward
            - self.boundary_penalty
            - self._reward_parameters.time_penalty
            + self._reward_parameters.terminal_reward * self._goal_reached
        )
    
        return reward
    
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
        die = torch.where(self._goal_reached > 0, ones, die)
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
        stats["linear_velocity_error"] += self.linear_velocity_error
        stats["boundary_dist"] += self.boundary_dist
        stats["energy_sum"] += self.energy_sum
        stats = self._task_parameters.boundary_penalty.update_statistics(stats)
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
        """

        self._goal_reached[env_ids] = 0
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
        p = torch.zeros((num_goals, 3), dtype=torch.float32, device=self._device)
        p[:, 2] = 2.0
        q = torch.zeros((num_goals, 4), dtype=torch.float32, device=self._device)
        q[:, 0] = 1
        # TODO: Get the target linear velocity from a sampler
        self._target_velocities[env_ids] = torch.rand((num_goals,), device=self._device) + 0.5 # [0.5, 1.5]
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
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        # Initial angle to define the position of the platform around the target
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        # theta = torch.zeros((num_resets,), device=self._device) # Fix initial positions
        initial_position = torch.zeros((num_resets, 3), device=self._device, dtype=torch.float32)
        initial_position[:, 0] = (r * torch.cos(theta) + self._target_positions[env_ids, 0])
        initial_position[:, 1] = (r * torch.sin(theta) + self._target_positions[env_ids, 1])

        # Computes the heading of the platform in the global frame to face the target
        target_position_local = (self._target_positions[env_ids, :2] - initial_position[:, :2])
        target_heading = torch.arctan2(target_position_local[:, 1], target_position_local[:, 0])

        # Randomize heading of the platform
        self._delta_headings[env_ids] = self._spawn_heading_sampler.sample(num_resets, step, device=self._device)
        theta = target_heading + self._delta_headings[env_ids]

        # Set the initial_orientation in quaternion
        initial_orientation = torch.zeros((num_resets, 4), device=self._device, dtype=torch.float32)
        initial_orientation[:, 0] = torch.cos(theta * 0.5)
        initial_orientation[:, 3] = torch.sin(theta * 0.5)

        # Randomizes the linear velocity of the platform
        initial_velocity = torch.zeros((num_resets, 6), device=self._device, dtype=torch.float32)
        linear_velocity = self._spawn_linear_velocity_sampler.sample(num_resets, step, device=self._device)
        # Rotate the linear velocity to the local frame
        initial_velocity[:, 0] = torch.cos(theta) * linear_velocity
        initial_velocity[:, 1] = torch.sin(theta) * linear_velocity

        # Randomizes the angular velocity of the platform
        angular_velocity = self._spawn_angular_velocity_sampler.sample(num_resets, step, device=self._device)
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



    def get_logs(self, step: int) -> dict:
        """
        Logs the task data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The task data.
        """

        dict = self._task_parameters.boundary_penalty.get_logs()
        if step % 50 == 1:
            dict = {**dict, **self.log_spawn_data(step)}
        return dict


    # Overload from Core to simplify observation space definition
    def define_observation_space(self):
        """
        Define the observation space dimensions.

        Args:
            dim_observation (int): Dimension of the observation.
            dim_task_data (int): Dimension of the task-specific data part of the observation.
        """
        dim_heading_error = 2
        dim_distance_error = 1
        dim_velocity_error = 1

        self._num_observations = (
            self._dim_velocity + self._dim_omega + dim_heading_error + dim_distance_error + dim_velocity_error
        )
        self._obs_buffer_len = self._task_parameters.obs_history_length

        self._obs_buffer_history = torch.zeros(
            (self._num_envs, self._obs_buffer_len, self._num_observations),
            device=self._device,
            dtype=torch.float32,
        )
        # Observation buffer must be flat
        self._obs_buffer = self._obs_buffer_history.view(self._num_envs, -1)

    # Overload from Core to simplify observation tensor update
    def update_observation_tensor(self, current_state: dict):
        """
        Update the observation tensor in the local frame.

        Args:
            current_state (dict): The current state of the system.

        Returns:
            None
        """

        # Orientation cos, sin of USV in the global frame
        cos_theta = current_state["orientation"][:, 0]
        sin_theta = current_state["orientation"][:, 1]

        # Transform linear velocity to the local frame
        lin_vel_global = current_state["linear_velocity"]
        self.lin_vel_local = torch.zeros_like(lin_vel_global)
        self.lin_vel_local[:, 0] = cos_theta * lin_vel_global[:, 0] + sin_theta * lin_vel_global[:, 1]
        self.lin_vel_local[:, 1] = -sin_theta * lin_vel_global[:, 0] + cos_theta * lin_vel_global[:, 1]

        # Transform position error to local frame
        position_error_global = self._target_positions - current_state["position"]
        position_error_local = torch.zeros_like(position_error_global)
        position_error_local[:, 0] = cos_theta * position_error_global[:, 0] + sin_theta * position_error_global[:, 1]
        position_error_local[:, 1] = -sin_theta * position_error_global[:, 0] + cos_theta * position_error_global[:, 1]
        self._position_error = position_error_local

        # Get target distance
        self.target_distance = torch.norm(self._position_error, dim=-1)

        # Get target bearing
        self.target_bearing = torch.zeros_like(self._position_error)
        self.target_bearing[:, 0] = self._position_error[:, 0] / self.target_distance
        self.target_bearing[:, 1] = self._position_error[:, 1] / self.target_distance
        # Get the angle in radians 
        self.target_bearing_rad = torch.atan2(self._position_error[:, 1], self._position_error[:, 0])

        # Get linear velocity error
        self.linear_velocity_error = self._target_velocities - self.lin_vel_local[:, 0]

        # Set the observation tensor
        # Shift the buffer
        self._obs_buffer_history[:,:-1,:] = self._obs_buffer_history[:,1:,:]
        # Include the new observation
        self._obs_buffer_history[:,-1, :2] = self.lin_vel_local # Linear velocity in local frame
        self._obs_buffer_history[:,-1, 2] = current_state["angular_velocity"] # Angular velocity
        self._obs_buffer_history[:,-1, 3:5] = self.target_bearing # Tartget bearing in local frame (cos, sin)
        self._obs_buffer_history[:,-1, 5] = self.target_distance # Target distance
        self._obs_buffer_history[:,-1, 6] = self.linear_velocity_error # Linear velocity error
        # Flatten the buffer
        self._obs_buffer = self._obs_buffer_history.view(self._num_envs, -1)
