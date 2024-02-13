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
    GoToPoseReward,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_task_parameters import (
    GoToPoseParameters,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.curriculum_helpers import (
    CurriculumSampler,
)

from omniisaacgymenvs.utils.arrow import VisualArrow
from omni.isaac.core.prims import XFormPrimView
from pxr import Usd

from typing import Tuple
import torch
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPoseTask(Core):
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

        super(GoToPoseTask, self).__init__(num_envs, device)
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
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self._target_headings = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 1

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
        if not "position_reward" in stats.keys():
            stats["position_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "heading_reward" in stats.keys():
            stats["heading_reward"] = torch_zeros()
        if not "heading_error" in stats.keys():
            stats["heading_error"] = torch_zeros()
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
        self._position_error = self._target_positions - current_state["position"]
        # heading distance
        heading = torch.arctan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        self._heading_error = torch.arctan2(
            torch.sin(self._target_headings - heading),
            torch.cos(self._target_headings - heading),
        )
        # Encode task data
        self._task_data[:, :2] = self._position_error
        self._task_data[:, 2] = torch.cos(self._heading_error)
        self._task_data[:, 3] = torch.sin(self._heading_error)
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
        self.heading_dist = torch.abs(self._heading_error)

        # Checks if the goal is reached
        position_goal_is_reached = (
            self.position_dist < self._task_parameters.position_tolerance
        ).int()
        heading_goal_is_reached = (
            self.heading_dist < self._task_parameters.heading_tolerance
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

        return self.position_reward + self.heading_reward

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
        stats["heading_reward"] += self.heading_reward
        stats["position_error"] += self.position_dist
        stats["heading_error"] += self.heading_dist
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
    ) -> list:
        """
        Generates a random goal for the task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            target_positions (torch.Tensor): The target positions.
            target_orientations (torch.Tensor): The target orientations.

        Returns:
            list: The target positions and orientations.
        """

        num_goals = len(env_ids)
        # Randomize position
        self._target_positions[env_ids] = (
            torch.rand((num_goals, 2), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        target_positions[env_ids, :2] += self._target_positions[env_ids]
        # Randomize heading
        self._target_headings[env_ids] = (
            torch.rand(num_goals, device=self._device) * math.pi * 2
        )
        target_orientations[env_ids, 0] = torch.cos(
            self._target_headings[env_ids] * 0.5
        )
        target_orientations[env_ids, 3] = torch.sin(
            self._target_headings[env_ids] * 0.5
        )
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
        r = self._spawn_position_sampler.sample(num_resets, step)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_position[:, 0] += (
            r * torch.cos(theta) + self._target_positions[env_ids, 0]
        )
        initial_position[:, 1] += (
            r * torch.sin(theta) + self._target_positions[env_ids, 1]
        )
        # Randomizes the heading of the platform
        initial_orientation = torch.zeros(
            (num_resets, 4), device=self._device, dtype=torch.float32
        )
        theta = (
            self._spawn_heading_sampler.sample(num_resets, step)
            + self._target_headings[env_ids]
        )
        initial_orientation[:, 0] = torch.cos(theta * 0.5)
        initial_orientation[:, 3] = torch.sin(theta * 0.5)
        # Randomizes the linear velocity of the platform
        initial_velocity = torch.zeros(
            (num_resets, 6), device=self._device, dtype=torch.float32
        )
        linear_velocity = self._spawn_linear_velocity_sampler.sample(num_resets, step)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[env_ids, 0] = linear_velocity * torch.cos(theta)
        initial_velocity[env_ids, 1] = linear_velocity * torch.sin(theta)
        # Randomizes the angular velocity of the platform
        angular_velocity = self._spawn_angular_velocity_sampler.sample(num_resets, step)
        initial_velocity[env_ids, 5] = angular_velocity
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
        body_radius = 0.1
        body_length = 0.5
        head_radius = 0.2
        head_length = 0.5
        poll_radius = 0.025
        poll_length = 2
        VisualArrow(
            prim_path=path + "/arrow",
            translation=position,
            name="target_0",
            body_radius=body_radius,
            body_length=body_length,
            poll_radius=poll_radius,
            poll_length=poll_length,
            head_radius=head_radius,
            head_length=head_length,
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

        arrows = XFormPrimView(prim_paths_expr="/World/envs/.*/arrow")
        scene.add(arrows)
        return scene, arrows
