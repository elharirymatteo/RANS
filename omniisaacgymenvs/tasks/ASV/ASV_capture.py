__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
__status__ = "development"

from omniisaacgymenvs.tasks.ASV.ASV_core import (
    Core,
    parse_data_dict,
)
from omniisaacgymenvs.tasks.ASV.ASV_task_rewards import (
    CaptureXYReward,
)
from omniisaacgymenvs.tasks.ASV.ASV_task_parameters import (
    CaptureXYParameters,
)
from omniisaacgymenvs.utils.pin import VisualPin

from omni.isaac.core.prims import XFormPrimView

import math
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class CaptureTask(Core):
    """
    Implements the CaptureXY task. The robot has to reach a target position."""

    def __init__(
        self,
        task_param: CaptureXYParameters,
        reward_param: CaptureXYReward,
        num_envs: int,
        device: str,
    ) -> None:
        super(CaptureTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = parse_data_dict(CaptureXYParameters(), task_param)
        self._reward_parameters = parse_data_dict(CaptureXYReward(), reward_param)

        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 0
        self.just_had_been_reset = torch.range(
            0, num_envs - 1, device=self._device, dtype=torch.long
        )

        # Initialize prev_position_dist with None
        self.prev_position_dist = None

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task."""

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )

        if not "distance_reward" in stats.keys():
            stats["distance_reward"] = torch_zeros()
        if not "alignment_reward" in stats.keys():
            stats["alignment_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "boundary_penalty" in stats.keys():
            stats["boundary_penalty"] = torch_zeros()
        if not "boundary_dist" in stats.keys():
            stats["boundary_dist"] = torch_zeros()
        return stats

    def get_state_observations(
        self, current_state: dict, observation_frame: str
    ) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot."""

        self._position_error = self._target_positions - current_state["position"]
        # Convert the orientation to radian
        theta = torch.atan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        # Compute the angle  of goal in global frame
        beta = torch.atan2(self._position_error[:, 1], self._position_error[:, 0])
        # Compute the angle error
        alpha = torch.fmod(beta - theta + math.pi, 2 * math.pi) - math.pi
        # Save the heading error, absolute of alpha
        self.heading_error = torch.abs(alpha)
        # Convert alpha to cos and sin
        self._task_data[:, 0] = torch.cos(alpha)
        self._task_data[:, 1] = torch.sin(alpha)
        # Compute the Distance to the goal
        self._task_data[:, 2] = torch.norm(self._position_error, dim=1)
        # Debug : print the task data
        # print(f"self._task_data: {self._task_data}")
        return self.update_observation_tensor(current_state, observation_frame)

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot."""

        # position error
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))

        self.boundary_dist = self.position_dist - self._task_parameters.kill_dist

        self.boundary_penalty = (
            -torch.exp(-self.boundary_dist / 0.25) * self._task_parameters.boundary_cost
        )

        # Checks if the goal is reached
        goal_is_reached = (
            self.position_dist < self._task_parameters.position_tolerance
        ).int()
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1
        # print out how many goals reached
        # print(f"goal_is_reached: {torch.sum(goal_is_reached)}")

        # If prev_position_dist is None, set it to position_dist
        if self.prev_position_dist is None:
            self.prev_position_dist = self.position_dist

        # Rewards
        self.distance_reward, self.alignment_reward = (
            self._reward_parameters.compute_reward(
                current_state,
                actions,
                self.position_dist,
                self.heading_error,
            )
        )

        self.distance_reward[self.just_had_been_reset] = 0
        self.just_had_been_reset = torch.tensor(
            [], device=self._device, dtype=torch.long
        )

        # Add reward for reaching the goal
        goal_reward = (self._goal_reached * self._task_parameters.goal_reward).float()

        # Save position_dist for next calculation, as prev_position_dist
        self.prev_position_dist = self.position_dist

        # print(f"distance_reward", self.distance_reward)
        # print(f"alignment_reward", self.alignment_reward)

        return (
            self.distance_reward
            + self.alignment_reward
            + goal_reward
            + self._task_parameters.time_reward
        )

    def update_kills(self, step) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not."""

        die = torch.zeros_like(self._goal_reached, dtype=torch.long)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)

        # Run curriculum if selected
        if self._task_parameters.spawn_curriculum:
            if step < self._task_parameters.spawn_curriculum_warmup:
                kill_dist = self._task_parameters.spawn_curriculum_kill_dist
            elif step > self._task_parameters.spawn_curriculum_end:
                kill_dist = self._task_parameters.kill_dist
            else:
                r = (step - self._task_parameters.spawn_curriculum_warmup) / (
                    self._task_parameters.spawn_curriculum_end
                    - self._task_parameters.spawn_curriculum_warmup
                )
                kill_dist = (
                    r
                    * (
                        self._task_parameters.kill_dist
                        - self._task_parameters.spawn_curriculum_kill_dist
                    )
                    + self._task_parameters.spawn_curriculum_kill_dist
                )
        else:
            kill_dist = self._task_parameters.kill_dist

        die = torch.where(self.position_dist > kill_dist, ones, die)
        die = torch.where(
            self._goal_reached
            >= self._task_parameters.kill_after_n_steps_in_tolerance,  # self._goal_reached > self._task_parameters.kill_after_n_steps_in_tolerance,
            ones,
            die,
        )
        return die

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics."""

        stats["distance_reward"] += self.distance_reward
        stats["alignment_reward"] += self.alignment_reward
        stats["position_error"] += self.position_dist
        stats["boundary_penalty"] += self.boundary_penalty
        stats["boundary_dist"] += self.boundary_dist
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task."""

        self._goal_reached[env_ids] = 0
        self.just_had_been_reset = env_ids.clone()

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
            torch.rand((num_goals, 2), device=self._device)
            * self._task_parameters.goal_random_position
            * 2
            - self._task_parameters.goal_random_position
        )
        targets_position[env_ids, :2] += self._target_positions[env_ids]
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

        # Debug : print the step
        # print(f"step: {step}")
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
        initial_position[env_ids, 0] += (r) * torch.cos(theta) + self._target_positions[
            env_ids, 0
        ]
        initial_position[env_ids, 1] += (r) * torch.sin(theta) + self._target_positions[
            env_ids, 1
        ]
        initial_position[env_ids, 2] += 0

        # Randomizes the heading of the platform
        random_orient = torch.rand(num_resets, device=self._device) * math.pi
        initial_orientation[env_ids, 0] = torch.cos(random_orient * 0.5)
        initial_orientation[env_ids, 3] = torch.sin(random_orient * 0.5)
        return initial_position, initial_orientation

    def generate_target(self, path, position):
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.
        A pin is generated to represent the 2D position to be reached by the agent."""

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

    def add_visual_marker_to_scene(self, scene):
        """
        Adds the visual marker to the scene."""

        pins = XFormPrimView(prim_paths_expr="/World/envs/.*/pin")
        scene.add(pins)
        return scene, pins
