__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_3DoF.curriculum_helpers import (
    CurriculumRateParameters,
)
from dataclasses import dataclass, field
from typing import Dict
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)

scaling_functions = {
    "linear": lambda x, p=0.0: x * p,
    "constant": lambda x, p=0.0: (x > 0) * p,
    "log": lambda x, p=0.0: torch.log(x + EPS),
    "exp": lambda x, p=0.0: torch.exp(x),
    "sqrt": lambda x, p=0.0: torch.sqrt(x),
    "square": lambda x, p=0.0: torch.pow(x, 2),
    "cube": lambda x, p=0.0: torch.pow(x, 3),
    "inv_exp": lambda x, p=1.0: torch.exp(-x / (p + EPS)),
}


@dataclass
class BasePenalty:
    """
    This class implements the base for all penalties
    """

    curriculum: CurriculumRateParameters = field(default_factory=dict)
    enable: bool = False

    def __post_init__(self):
        self.curriculum = CurriculumRateParameters(**self.curriculum)
        self.name = "".join(
            [
                "_" + c.lower() if (c.isupper() and i != 0) else c.lower()
                for i, c in enumerate(type(self).__name__)
            ]
        )
        self.last_rate = None
        self.last_penalties = None

    def get_rate(self, step: int) -> float:
        """
        Gets the difficulty for the given step.

        Args:
            step (int): Current step.

        Returns:
            float: Current difficulty.
        """

        return self.curriculum.get(step)

    def compute_penalty(self, value, step):
        raise NotImplementedError

    def get_unweigthed_penalties(self):
        if self.last_rate is not None:
            return self.last_penalties

    def get_last_rate(self) -> float:
        if self.last_rate is not None:
            return self.last_rate


@dataclass
class EnergyPenalty(BasePenalty):
    """
    This class has access to the actions and applies a penalty based on how many actions are taken.
    """

    weight: float = 0.1

    def __post_init__(self):
        super().__post_init__()
        assert self.weight > 0, "Weight must be positive"

    def compute_penalty(
        self, state: Dict["str", torch.Tensor], actions: torch.Tensor, step: int
    ):
        """
        Computes the penalty based on the number of actions taken.

        Args:
            state (Dict[str, torch.Tensor]): State of the system.
            actions (torch.Tensor): Actions taken.
            step (int): Current step.

        Returns:
            torch.Tensor: Penalty.
        """

        if self.enable:
            self.last_rate = self.get_rate(step)
            self.last_penalties = torch.sum(torch.abs(actions), -1)
            return self.last_penalties * self.last_rate * self.weight
        else:
            return torch.zeros(
                [actions.shape[0]], dtype=torch.float32, device=actions.device
            )


@dataclass
class LinearVelocityPenalty(BasePenalty):
    """
    This class has access to the linear velocity and applies a penalty based on its norm.
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
        Computes the penalty based on the norm of the linear velocity.

        Args:
            state (Dict[str, torch.Tensor]): State of the system.
            actions (torch.Tensor): Actions taken.
            step (int): Current step.

        Returns:
            torch.Tensor: Penalty.
        """

        if self.enable:
            self.last_rate = self.get_rate(step)
            # compute the norm of the linear velocity
            norm = torch.norm(state["linear_velocity"], dim=-1) - self.min_value
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
            norm = torch.abs(state["angular_velocity"]) - self.min_value
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


@dataclass
class BoundaryPenalty(BasePenalty):
    """
    This class has access to the state and applies a penalty based on the distance to the boundaries.
    """

    weight: float = 10.0
    scaling_function: str = "inv_exp"
    scaling_parameter: float = 0.5
    saturation_value: float = 2.0

    def __post_init__(self):
        super().__post_init__()
        assert self.weight > 0, "Weight must be positive"
        assert self.scaling_function in scaling_functions, "Scaling function not found"
        assert self.saturation_value > 0, "Saturation value must be positive"

        self.scaling_function = scaling_functions[self.scaling_function]

    def compute_penalty(self, distance, step: int):
        """
        Computes the penalty based on the distance to the boundaries.

        Args:
            state (Dict[str, torch.Tensor]): State of the system.
            step (int): Current step.

        Returns:
            torch.Tensor: Penalty.
        """

        self.last_rate = self.get_rate(step)
        self.last_penalties = torch.clamp(
            self.scaling_function(distance, self.scaling_parameter),
            0,
            self.saturation_value,
        )
        return self.last_penalties * self.last_rate * self.weight

    def get_stats_name(self) -> list:
        """
        Returns the names of the statistics to be computed.

        Returns:
            list: Names of the statistics to be tracked.
        """

        return ["penalties/" + self.name]

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict): Current statistics.

        Returns:
            dict: Updated statistics.
        """

        stats["penalties/" + self.name] += self.get_unweigthed_penalties()
        return stats

    def get_logs(self) -> dict:
        """
        Logs the penalty.

        Returns:
            dict: Dictionary containing the penalty.
        """

        return {
            "penalties/" + self.name + "_weight": self.get_last_rate() * self.weight
        }


@dataclass
class ContactPenalty(BasePenalty):
    """
    This class has access to the state and applies a penalty based on the collision.
    """

    weight: float = 10.0
    scaling_function: str = "constant"
    scaling_parameter: float = 1.0
    kill_on_contact: bool = False
    kill_threshold: float = 1.0
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

    def compute_penalty(self, forces: torch.Tensor, step: int):
        """
        Computes the penalty based on the norm of the angular velocity.

        Args:
            forces (torch.Tensor): The contact forces.
            step (int): Current step.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Penalty, kills.
        """
        self.last_rate = self.get_rate(step)
        if self.enable:
            # computes the threshold for the contact forces
            norm = forces - self.min_value
            # apply ranging function
            norm[norm < EPS] = 0
            norm[norm > (self.max_value - self.min_value)] = (
                self.max_value - self.min_value
            )
            # apply scaling function
            norm = self.scaling_function(norm, p=self.scaling_parameter)
            self.last_penalties = norm
            # kill the episode if the contact forces are too high
            if self.kill_on_contact:
                kills = forces > self.kill_threshold
            else:
                kills = torch.zeros(
                    [norm.shape[0]], dtype=torch.float32, device=norm.device
                )
            return norm * self.last_rate * self.weight, kills
        else:
            self.last_penalties = torch.zeros(
                [forces.shape[0]], dtype=torch.float32, device=forces.device
            )
            return torch.zeros(
                [forces.shape[0]], dtype=torch.float32, device=forces.device
            ), torch.zeros([forces.shape[0]], dtype=torch.bool, device=forces.device)

    def get_stats_name(self) -> list:
        """
        Returns the names of the statistics to be computed.

        Returns:
            list: Names of the statistics to be tracked.
        """

        return ["penalties/" + self.name]

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict): Current statistics.

        Returns:
            dict: Updated statistics.
        """

        stats["penalties/" + self.name] += self.get_unweigthed_penalties()
        return stats

    def get_logs(self) -> dict:
        """
        Logs the penalty.

        Returns:
            dict: Dictionary containing the penalty.
        """

        return {
            "penalties/" + self.name + "_weight": self.get_last_rate() * self.weight
        }


@dataclass
class ConeShapePenalty(BasePenalty):
    """
    This class has access to the realtive angle of FP to dock and applies a penalty based on its norm.
    """

    weight: float = 0.1
    scaling_function: str = "linear"
    scaling_parameter: float = 1.0
    min_value: float = 0.1745  # pi/18=10deg
    max_value: float = 3.1415

    def __post_init__(self):
        super().__post_init__()
        assert self.weight > 0, "Weight must be positive"
        assert self.scaling_function in scaling_functions, "Scaling function not found"
        assert (
            self.min_value < self.max_value
        ), "Min value must be smaller than max value"

        self.scaling_function = scaling_functions[self.scaling_function]

    def compute_penalty(self, relative_angle: torch.Tensor, step: int):
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
            norm = torch.abs(relative_angle) - self.min_value
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
                [relative_angle.shape[0]],
                dtype=torch.float32,
                device=relative_angle.device,
            )

    def get_stats_name(self) -> list:
        """
        Returns the names of the statistics to be computed.

        Returns:
            list: Names of the statistics to be tracked.
        """

        return ["penalties/" + self.name]

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict): Current statistics.

        Returns:
            dict: Updated statistics.
        """

        stats["penalties/" + self.name] += self.get_unweigthed_penalties()
        return stats

    def get_logs(self) -> dict:
        """
        Logs the penalty.

        Returns:
            dict: Dictionary containing the penalty.
        """

        return {
            "penalties/" + self.name + "_weight": self.get_last_rate() * self.weight
        }
