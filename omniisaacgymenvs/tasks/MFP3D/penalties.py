__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_3DoF.penalties import (
    BasePenalty,
    EnergyPenalty,
    LinearVelocityPenalty,
    scaling_functions,
    BoundaryPenalty,
)

from dataclasses import dataclass, field
from typing import Dict
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class AngularVelocityPenalty(BasePenalty):
    """
    This class has access to the angular velocity and applies a penalty based on its norm.
    """

    weight: float = 0.1
    scaling_function: str = "linear"
    scaling_parameter: float = 1.0
    min_value: float = 0
    max_value: float = float("inf")

    def __post_init__(self):
        super().__post_init__()
        assert self.weight > 0, "Weight must be positive"
        assert self.scaling_function in scaling_functions, "Scaling function not found"
        assert (
            self.min_value < self.max_value
        ), "Min value must be smaller than max value"

        self.scaling_function = scaling_functions[self.scaling_function]

    def compute_penalty(
        self, state: Dict["str", torch.Tensor], actions: torch.Tensor, step: int
    ):
        """
        Computes the penalty based on the norm of the angular velocity.

        Args:
            state (Dict[str, torch.Tensor]): State of the system.
            actions (torch.Tensor): Actions taken.
            step (int): Current step.

        Returns:
            torch.Tensor: Penalty.
        """

        if self.enable:
            self.last_rate = self.get_rate(step)
            # compute the norm of the angular velocity
            norm = torch.norm(state["angular_velocity"], dim=-1) - self.min_value
            # apply ranging function
            norm[norm < 0] = 0
            norm[norm > (self.max_value - self.min_value)] = (
                self.max_value - self.min_value
            )
            # apply scaling function
            norm = self.scaling_function(norm, p=self.scaling_parameter)
            self.last_penalties = norm
            return norm * self.last_rate * self.weight
        else:
            return torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )


penalty_classes = {
    "energy_penalty": EnergyPenalty,
    "linear_velocity_penalty": LinearVelocityPenalty,
    "angular_velocity_penalty": AngularVelocityPenalty,
}


@dataclass
class EnvironmentPenalties:
    energy_penalty: EnergyPenalty = field(default_factory=dict)
    linear_velocity_penalty: LinearVelocityPenalty = field(default_factory=dict)
    angular_velocity_penalty: AngularVelocityPenalty = field(default_factory=dict)

    def __post_init__(self):
        self.penalties = []
        self.energy_penalty = EnergyPenalty(**self.energy_penalty)
        if self.energy_penalty.enable:
            self.penalties.append(self.energy_penalty)
        self.linear_velocity_penalty = LinearVelocityPenalty(
            **self.linear_velocity_penalty
        )
        if self.linear_velocity_penalty.enable:
            self.penalties.append(self.linear_velocity_penalty)
        self.angular_velocity_penalty = AngularVelocityPenalty(
            **self.angular_velocity_penalty
        )
        if self.angular_velocity_penalty.enable:
            self.penalties.append(self.angular_velocity_penalty)

    def compute_penalty(
        self, state: Dict[str, torch.Tensor], actions: torch.Tensor, step: int
    ) -> torch.Tensor:
        """
        Computes the penalties.

        Args:
            state (Dict[str, torch.Tensor]): State of the system.
            actions (torch.Tensor): Actions taken.
            step (int): Current step.

        Returns:
            torch.Tensor: Penalty.
        """

        penalties = torch.zeros(
            [actions.shape[0]], dtype=torch.float32, device=actions.device
        )
        for penalty in self.penalties:
            penalties += penalty.compute_penalty(state, actions, step)
        return penalties

    def get_stats_name(self) -> list:
        """
        Returns the names of the statistics to be computed.

        Returns:
            list: Names of the statistics to be tracked.
        """

        names = []
        for penalty in self.penalties:
            names.append("penalties/" + penalty.name)
        return names

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict): Current statistics.

        Returns:
            dict: Updated statistics.
        """

        for penalty in self.penalties:
            stats["penalties/" + penalty.name] += penalty.get_unweigthed_penalties()
        return stats

    def get_logs(self) -> dict:
        """
        Logs the penalty.

        Returns:
            dict: Dictionary containing the penalty.
        """

        dict = {}
        for penalty in self.penalties:
            dict["penalties/" + penalty.name + "_weight"] = (
                penalty.get_last_rate() * penalty.weight
            )
        return dict
