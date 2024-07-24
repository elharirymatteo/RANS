__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Luis Batista"
__email__ = "luis.batista@gatech.edu"
__status__ = "development"

import torch
from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class GoThroughPositionReward:
    """
    Reward function and parameters for the GoThroughXY task."""

    name: str = "GoThroughPosition"
    heading_reward_mode: str = "linear"
    velocity_reward_mode: str = "linear"
    heading_exponential_reward_coeff: float = 0.25
    velocity_exponential_reward_coeff: float = 0.25
    time_penalty: float = 0.0
    terminal_reward: float = 0.0
    dt: float = 0.02
    action_repeat: int = 10
    position_scale: float = 1.0
    heading_scale: float = 1.0
    velocity_scale: float = 1.0

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""


        assert self.velocity_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."
        assert self.heading_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

        self.dt = self.dt * self.action_repeat

    def compute_reward(
        self,
        current_state,
        actions: torch.Tensor,
        position_progress: torch.Tensor,
        heading_error: torch.Tensor,
        velocity_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the GoToPose task."""

        position_reward = self.position_scale * position_progress / self.dt

        if self.heading_reward_mode.lower() == "linear":
            heading_reward = 1.0 / (1.0 + heading_error) * self.heading_scale
        elif self.heading_reward_mode.lower() == "square":
            heading_reward = 1.0 / (1.0 + heading_error) * self.heading_scale
        elif self.heading_reward_mode.lower() == "exponential":
            heading_reward = (
                torch.exp(-heading_error / self.heading_exponential_reward_coeff)
                * self.heading_scale
            )
        else:
            raise ValueError("Unknown reward type.")

        if self.velocity_reward_mode.lower() == "linear":
            velocity_reward = 1.0 / (1.0 + heading_error) * self.velocity_scale
        elif self.velocity_reward_mode.lower() == "square":
            velocity_reward = 1.0 / (1.0 + heading_error) * self.velocity_scale
        elif self.velocity_reward_mode.lower() == "exponential":
            velocity_reward = (
                torch.exp(-velocity_error / self.velocity_exponential_reward_coeff)
                * self.velocity_scale
            )
        else:
            raise ValueError("Unknown reward type.")

        return position_reward, heading_reward, velocity_reward

