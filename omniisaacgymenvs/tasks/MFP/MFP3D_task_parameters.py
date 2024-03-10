__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from dataclasses import dataclass, field
from omniisaacgymenvs.tasks.MFP.curriculum_helpers import (
    CurriculumParameters,
)
from omniisaacgymenvs.tasks.MFP.MFP2D_penalties import (
    BoundaryPenalty,
)

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class GoToXYZParameters:
    """
    Parameters for the GoToXY task.
    """

    name: str = "GoToXYZ"
    position_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_position: float = 0.0
    kill_dist: float = 10.0
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.spawn_position_curriculum = CurriculumParameters(
            **self.spawn_position_curriculum
        )
        self.spawn_linear_velocity_curriculum = CurriculumParameters(
            **self.spawn_linear_velocity_curriculum
        )
        self.spawn_angular_velocity_curriculum = CurriculumParameters(
            **self.spawn_angular_velocity_curriculum
        )


@dataclass
class GoToPoseParameters:
    """
    Parameters for the GoToPose task.
    """

    name: str = "GoToPose"
    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_position: float = 0.0
    kill_dist: float = 10.0

    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.position_tolerance > 0, "Position tolerance must be positive."
        assert self.orientation_tolerance > 0, "Heading tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.spawn_position_curriculum = CurriculumParameters(
            **self.spawn_position_curriculum
        )
        self.spawn_heading_curriculum = CurriculumParameters(
            **self.spawn_heading_curriculum
        )
        self.spawn_linear_velocity_curriculum = CurriculumParameters(
            **self.spawn_linear_velocity_curriculum
        )
        self.spawn_angular_velocity_curriculum = CurriculumParameters(
            **self.spawn_angular_velocity_curriculum
        )


@dataclass
class TrackXYZVelocityParameters:
    """
    Parameters for the TrackXYVelocity task.
    """

    name: str = "TrackXYZVelocity"
    lin_vel_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_velocity: float = 0.75
    kill_dist: float = 500.0

    target_linear_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.lin_vel_tolerance > 0, "Linear velocity tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_velocity >= 0, "Goal random velocity must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.target_linear_velocity_curriculum = CurriculumParameters(
            **self.target_linear_velocity_curriculum
        )
        self.spawn_linear_velocity_curriculum = CurriculumParameters(
            **self.spawn_linear_velocity_curriculum
        )
        self.spawn_angular_velocity_curriculum = CurriculumParameters(
            **self.spawn_angular_velocity_curriculum
        )


@dataclass
class Track6DoFVelocityParameters:
    """
    Parameters for the TrackXYOVelocity task.
    """

    name: str = "Track6DoFVelocity"
    lin_vel_tolerance: float = 0.01
    ang_vel_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_linear_velocity: float = 0.75
    goal_random_angular_velocity: float = 1
    kill_dist: float = 500.0

    target_linear_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    target_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.lin_vel_tolerance > 0, "Linear velocity tolerance must be positive."
        assert (
            self.ang_vel_tolerance > 0
        ), "Angular velocity tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert (
            self.goal_random_linear_velocity >= 0
        ), "Goal random linear velocity must be positive."
        assert (
            self.goal_random_angular_velocity >= 0
        ), "Goal random angular velocity must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.target_linear_velocity_curriculum = CurriculumParameters(
            **self.target_linear_velocity_curriculum
        )
        self.target_angular_velocity_curriculum = CurriculumParameters(
            **self.target_angular_velocity_curriculum
        )
        self.spawn_linear_velocity_curriculum = CurriculumParameters(
            **self.spawn_linear_velocity_curriculum
        )
        self.spawn_angular_velocity_curriculum = CurriculumParameters(
            **self.spawn_angular_velocity_curriculum
        )
