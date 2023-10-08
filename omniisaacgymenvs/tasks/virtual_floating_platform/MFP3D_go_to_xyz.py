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
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_rewards import (
    GoToXYZReward,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP3D_task_parameters import (
    GoToXYZParameters,
)
from omniisaacgymenvs.utils.pin3D import VisualPin3D

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_go_to_xy import (
    GoToXYTask as GoToXYTask2D,
)

from omni.isaac.core.prims import XFormPrimView

import math
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class GoToXYZTask(GoToXYTask2D, Core):
    """
    Implements the GoToXY task. The robot has to reach a target position."""

    def __init__(
        self,
        task_param: GoToXYZParameters,
        reward_param: GoToXYZReward,
        num_envs: int,
        device: str,
    ) -> None:
        Core.__init__(self, num_envs, device)
        # Task and reward parameters
        self._task_parameters = parse_data_dict(GoToXYZParameters(), task_param)
        self._reward_parameters = parse_data_dict(GoToXYZReward(), reward_param)

        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 0

    def update_observation_tensor(self, current_state: dict) -> torch.Tensor:
        return Core.update_observation_tensor(self, current_state)

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot."""

        self._position_error = self._target_positions - current_state["position"]
        self._task_data[:, :3] = self._position_error
        return self.update_observation_tensor(current_state)

    def get_goals(
        self,
        env_ids: torch.Tensor,
        targets_position: torch.Tensor,
        targets_orientation: torch.Tensor,
    ) -> list:
        """
        Generates a random goal for the task."""

        num_goals = len(env_ids)
        self._target_positions[env_ids] = (
            torch.rand((num_goals, 3), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        targets_position[env_ids, :3] += self._target_positions[env_ids]
        return targets_position, targets_orientation

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

    def add_visual_marker_to_scene(self, scene):
        """
        Adds the visual marker to the scene."""

        pins = XFormPrimView(prim_paths_expr="/World/envs/.*/pin")
        scene.add(pins)
        return scene, pins
