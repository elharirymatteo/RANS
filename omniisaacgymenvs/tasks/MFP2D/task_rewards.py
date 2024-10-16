__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import torch
from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)

@dataclass
class TrackLinearAngularVelocityReward:
    """
    Reward function and parameters for the TrackXYOVelocity task.
    """

    name: str = "TrackLinearAngularVelocity"
    linear_reward_mode: str = "linear"
    angular_reward_mode: str = "linear"
    linear_exponential_reward_coeff: float = 0.25
    angular_exponential_reward_coeff: float = 0.25
    linear_scale: float = 1.0
    angular_scale: float = 1.0

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.linear_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."
        assert self.angular_reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state,
        actions: torch.Tensor,
        linear_velocity_error: torch.Tensor,
        angular_velocity_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the TrackXYOVelocity task.
        """

        if self.linear_reward_mode.lower() == "linear":
            linear_reward = 1.0 / (1.0 + linear_velocity_error) * self.linear_scale
        elif self.linear_reward_mode.lower() == "square":
            linear_reward = 1.0 / (1.0 + linear_velocity_error) * self.linear_scale
        elif self.linear_reward_mode.lower() == "exponential":
            linear_reward = (
                torch.exp(-linear_velocity_error / self.linear_exponential_reward_coeff)
                * self.linear_scale
            )
        else:
            raise ValueError("Unknown reward type.")

        if self.angular_reward_mode.lower() == "linear":
            angular_reward = 1.0 / (1.0 + angular_velocity_error) * self.angular_scale
        elif self.angular_reward_mode.lower() == "square":
            angular_reward = 1.0 / (1.0 + angular_velocity_error) * self.angular_scale
        elif self.angular_reward_mode.lower() == "exponential":
            angular_reward = (
                torch.exp(
                    -angular_velocity_error / self.angular_exponential_reward_coeff
                )
                * self.angular_scale
            )
        else:
            raise ValueError("Unknown reward type.")
        return linear_reward, angular_reward


@dataclass
class TrackLinearVelocityHeadingReward:
    """
    Reward function and parameters for the TrackXYVelHeading task."""

    name: str = "TrackLinearVelocityHeading"
    velocity_reward_mode: str = "linear"
    heading_reward_mode: str = "linear"
    velocity_exponential_reward_coeff: float = 0.25
    heading_exponential_reward_coeff: float = 0.25
    velocity_scale: float = 1.0
    heading_scale: float = 1.0

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

    def compute_reward(
        self,
        current_state,
        actions: torch.Tensor,
        velocity_error: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the GoToPose task."""

        if self.velocity_reward_mode.lower() == "linear":
            velocity_reward = 1.0 / (1.0 + velocity_error) * self.velocity_scale
        elif self.velocity_reward_mode.lower() == "square":
            velocity_reward = 1.0 / (1.0 + velocity_error) * self.velocity_scale
        elif self.velocity_reward_mode.lower() == "exponential":
            velocity_reward = (
                torch.exp(-velocity_error / self.velocity_exponential_reward_coeff)
                * self.velocity_scale
            )
        else:
            raise ValueError("Unknown reward type.")

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
        return velocity_reward, heading_reward
