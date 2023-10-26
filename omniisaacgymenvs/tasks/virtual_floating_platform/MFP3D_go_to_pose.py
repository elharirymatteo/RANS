__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_core import (
    Core,
    parse_data_dict,
    quat_to_mat,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_rewards import (
    GoToPoseReward,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_parameters import (
    GoToPoseParameters,
)
from omniisaacgymenvs.utils.arrow3D import VisualArrow3D

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_go_to_pose import (
    GoToPoseTask as GoToPoseTask2D,
)

from omni.isaac.core.prims import XFormPrimView

import math
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToPoseTask(GoToPoseTask2D, Core):
    """
    Implements the GoToPose task. The robot has to reach a target position and heading.
    """

    def __init__(
        self,
        task_param: GoToPoseParameters,
        reward_param: GoToPoseReward,
        num_envs: int,
        device: str,
    ) -> None:
        Core.__init__(self, num_envs, device)
        # Task and reward parameters
        self._task_parameters = parse_data_dict(GoToPoseParameters(), task_param)
        self._reward_parameters = parse_data_dict(GoToPoseReward(), reward_param)

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
        # self._target_headings_as_mat = torch.zeros((self._num_envs, 3, 3), device=self._device, dtype=torch.float32)
        self._task_label = self._task_label * 1

    def update_observation_tensor(self, current_state: dict) -> torch.Tensor:
        return Core.update_observation_tensor(self, current_state)

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.""" ""

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
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot."""

        # position error
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        trace = (
            self._heading_error[:, 0, 0]
            + self._heading_error[:, 1, 1]
            + self._heading_error[:, 2, 2]
        )
        self.heading_dist = torch.arccos((trace - 1) / 2)

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
        return self.position_reward + self.heading_reward

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
    ) -> list:
        """
        Generates a random goal for the task."""

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
        quat[env_ids, 0] = torch.sqrt(uvw[:, 0]) * torch.cos(uvw[:, 2] * 2 * math.pi)
        quat[env_ids, 1] = torch.sqrt(1 - uvw[:, 0]) * torch.sin(
            uvw[:, 1] * 2 * math.pi
        )
        quat[env_ids, 2] = torch.sqrt(1 - uvw[:, 0]) * torch.cos(
            uvw[:, 1] * 2 * math.pi
        )
        quat[env_ids, 3] = torch.sqrt(uvw[:, 0]) * torch.sin(uvw[:, 2] * 2 * math.pi)
        # cast quaternions to rotation matrix
        self._target_headings = quat_to_mat(quat)
        return target_positions, quat

    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        Generates spawning positions for the robots following a curriculum."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self._goal_reached[env_ids] = 0
        # Run curriculum if selected
        if self._task_parameters.spawn_curriculum:
            if step < self._task_parameters.spawn_curriculum_warmup:
                rmax = self._task_parameters.spawn_curriculum_max_dist
                rmin = self._task_parameters.spawn_curriculum_min_dist
            elif step > self._task_parameters.spawn_curriculum_end:
                rmax = self._task_parameters.max_spawn_dist
                rmin = self._task_parameters.min_spawn_dist
            else:
                r = (step - self._task_parameters.spawn_curriculum_warmup) / (
                    self._task_parameters.spawn_curriculum_end
                    - self._task_parameters.spawn_curriculum_warmup
                )
                rmax = (
                    r
                    * (
                        self._task_parameters.max_spawn_dist
                        - self._task_parameters.spawn_curriculum_max_dist
                    )
                    + self._task_parameters.spawn_curriculum_max_dist
                )
                rmin = (
                    r
                    * (
                        self._task_parameters.min_spawn_dist
                        - self._task_parameters.spawn_curriculum_min_dist
                    )
                    + self._task_parameters.spawn_curriculum_min_dist
                )
        else:
            rmax = self._task_parameters.max_spawn_dist
            rmin = self._task_parameters.min_spawn_dist

        # Randomizes the starting position of the platform
        r = torch.rand((num_resets,), device=self._device) * (rmax - rmin) + rmin
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        phi = torch.rand((num_resets,), device=self._device) * math.pi
        initial_position[env_ids, 0] += (r) * torch.cos(theta) + self._target_positions[
            env_ids, 0
        ]
        initial_position[env_ids, 1] += (r) * torch.sin(theta) + self._target_positions[
            env_ids, 1
        ]
        initial_position[env_ids, 2] += (r) * torch.cos(phi) + self._target_positions[
            env_ids, 2
        ]

        # Randomizes the orientation of the platform
        uvw = torch.rand((num_resets, 3), device=self._device)
        initial_orientation[env_ids, 0] = torch.sqrt(uvw[:, 0]) * torch.cos(
            uvw[:, 2] * 2 * math.pi
        )
        initial_orientation[env_ids, 1] = torch.sqrt(1 - uvw[:, 0]) * torch.sin(
            uvw[:, 1] * 2 * math.pi
        )
        initial_orientation[env_ids, 2] = torch.sqrt(1 - uvw[:, 0]) * torch.cos(
            uvw[:, 1] * 2 * math.pi
        )
        initial_orientation[env_ids, 3] = torch.sqrt(uvw[:, 0]) * torch.sin(
            uvw[:, 2] * 2 * math.pi
        )
        return initial_position, initial_orientation

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
