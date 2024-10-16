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


@dataclass
class GoThroughPositionSequenceReward:
    """
    Reward function and parameters for the GoThroughXYSequence task."""

    name: str = "GoThroughPositionSequence"
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


@dataclass
class GoThroughPoseReward:
    """
    Reward function and parameters for the GoThroughXY task."""

    name: str = "GoThroughPose"
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


@dataclass
class GoThroughPoseSequenceReward:
    """
    Reward function and parameters for the GoThroughXYSequence task."""

    name: str = "GoThroughPoseSequence"
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


@dataclass
class GoThroughGateReward:
    """
    Reward function and parameters for the GoThroughXY task."""

    name: str = "GoThroughGate"
    heading_reward_mode: str = "linear"
    heading_exponential_reward_coeff: float = 0.25
    time_penalty: float = 0.0
    terminal_reward: float = 0.0
    reverse_penalty: float = 0.0
    dt: float = 0.02
    action_repeat: int = 10
    position_scale: float = 1.0
    heading_scale: float = 1.0

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

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

        return position_reward, heading_reward


@dataclass
class GoThroughGateSequenceReward:
    """
    Reward function and parameters for the GoThroughXY task."""

    name: str = "GoThroughGate"
    heading_reward_mode: str = "linear"
    heading_exponential_reward_coeff: float = 0.25
    time_penalty: float = 0.0
    terminal_reward: float = 0.0
    reverse_penalty: float = 0.0
    dt: float = 0.02
    action_repeat: int = 10
    position_scale: float = 1.0
    heading_scale: float = 1.0

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

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

        return position_reward, heading_reward


@dataclass
class GoToPositionReward:
    """ "
    Reward function and parameters for the GoToXY task."""

    name: str = "GoToPosition"
    reward_mode: str = "linear"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        position_error: torch.Tensor,
    ) -> torch.Tensor:
        """
        Defines the function used to compute the reward for the GoToXY task."""

        if self.reward_mode.lower() == "linear":
            position_reward = 1.0 / (1.0 + position_error)
        elif self.reward_mode.lower() == "square":
            position_reward = 1.0 / (1.0 + position_error * position_error)
        elif self.reward_mode.lower() == "exponential":
            position_reward = torch.exp(-position_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return position_reward


@dataclass
class GoToPoseReward:
    """
    Reward function and parameters for the GoToPose task."""

    name: str = "GoToPose"
    position_reward_mode: str = "linear"
    heading_reward_mode: str = "linear"
    position_exponential_reward_coeff: float = 0.25
    heading_exponential_reward_coeff: float = 0.25
    position_scale: float = 1.0
    heading_scale: float = 1.0

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.position_reward_mode.lower() in [
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
        position_error: torch.Tensor,
        heading_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the GoToPose task."""

        if self.position_reward_mode.lower() == "linear":
            position_reward = 1.0 / (1.0 + position_error) * self.position_scale
        elif self.position_reward_mode.lower() == "square":
            position_reward = 1.0 / (1.0 + position_error) * self.position_scale
        elif self.position_reward_mode.lower() == "exponential":
            position_reward = (
                torch.exp(-position_error / self.position_exponential_reward_coeff)
                * self.position_scale
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
        return position_reward, heading_reward


@dataclass
class TrackLinearVelocityReward:
    """
    Reward function and parameters for the TrackXYVelocity task."""

    name: str = "TrackLinearVelocity"
    reward_mode: str = "linear"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self) -> None:
        """
        Checks that the reward parameters are valid."""

        assert self.reward_mode.lower() in [
            "linear",
            "square",
            "exponential",
        ], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(
        self,
        current_state: torch.Tensor,
        actions: torch.Tensor,
        velocity_error: torch.Tensor,
    ) -> None:
        """
        Defines the function used to compute the reward for the TrackXYVelocity task."""

        if self.reward_mode.lower() == "linear":
            velocity_reward = 1.0 / (1.0 + velocity_error)
        elif self.reward_mode.lower() == "square":
            velocity_reward = 1.0 / (1.0 + velocity_error * velocity_error)
        elif self.reward_mode.lower() == "exponential":
            velocity_reward = torch.exp(-velocity_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return velocity_reward


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
