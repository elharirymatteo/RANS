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
from omniisaacgymenvs.tasks.common_3DoF.curriculum_helpers import (
    CurriculumParameters,
)
from omniisaacgymenvs.tasks.common_3DoF.penalties import (
    BoundaryPenalty,
    ConeShapePenalty,
    ContactPenalty,
)

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class GoToPositionParameters:
    """
    Parameters for the GoToPosition task.
    """

    name: str = "GoToPosition"
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
        assert self.position_tolerance > 0, "Position tolerance must be positive."
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
    heading_tolerance: float = 0.025
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
        assert self.heading_tolerance > 0, "Heading tolerance must be positive."
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
class GoThroughPositionParameters:
    """
    Parameters for the GoThroughPosition task.
    """

    name: str = "GoThroughPosition"
    position_tolerance: float = 0.1
    linear_velocity_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    kill_dist: float = 10.0
    reference_frame: str = "world"

    target_linear_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.position_tolerance > 0, "Position tolerance must be positive."
        assert (
            self.linear_velocity_tolerance > 0
        ), "Velocity tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.target_linear_velocity_curriculum = CurriculumParameters(
            **self.target_linear_velocity_curriculum
        )
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
class GoThroughPositionSequenceParameters:
    """
    Parameters for the GoThroughPositionSequence task.
    """

    name: str = "GoThroughPositionSequence"
    position_tolerance: float = 0.1
    linear_velocity_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    kill_dist: float = 10.0
    num_points: int = 5

    target_linear_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.position_tolerance > 0, "Position tolerance must be positive."
        assert (
            self.linear_velocity_tolerance > 0
        ), "Velocity tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."
        assert self.num_points > 0, "Number of points must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.target_linear_velocity_curriculum = CurriculumParameters(
            **self.target_linear_velocity_curriculum
        )
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
class GoThroughPoseParameters:
    """
    Parameters for the GoThroughPose task.
    """

    name: str = "GoThroughPose"
    position_tolerance: float = 0.1
    heading_tolerance: float = 0.05
    linear_velocity_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    kill_dist: float = 10.0

    target_linear_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.position_tolerance > 0, "Position tolerance must be positive."
        assert self.heading_tolerance > 0, "Heading tolerance must be positive."
        assert (
            self.linear_velocity_tolerance > 0
        ), "Velocity tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.target_linear_velocity_curriculum = CurriculumParameters(
            **self.target_linear_velocity_curriculum
        )
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
class GoThroughPoseSequenceParameters:
    """
    Parameters for the GoThroughPoseSequence task.
    """

    name: str = "GoThroughPoseSequence"
    position_tolerance: float = 0.1
    heading_tolerance: float = 0.05
    linear_velocity_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 1
    goal_random_position: float = 0.0
    kill_dist: float = 10.0
    num_points: int = 5

    target_linear_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.position_tolerance > 0, "Position tolerance must be positive."
        assert self.heading_tolerance > 0, "Heading tolerance must be positive."
        assert (
            self.linear_velocity_tolerance > 0
        ), "Velocity tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."
        assert self.num_points > 0, "Number of points must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.target_linear_velocity_curriculum = CurriculumParameters(
            **self.target_linear_velocity_curriculum
        )
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
class GoThroughGateParameters:
    """
    Parameters for the GoThroughGate task.
    """

    name: str = "GoThroughGate"
    goal_random_position: float = 0.0
    kill_dist: float = 10.0
    gate_width: float = 1.5
    gate_thickness: float = 0.2

    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    contact_penalty: ContactPenalty = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.gate_width > 0, "Gate width must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.contact_penalty = ContactPenalty(**self.contact_penalty)
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
class GoThroughGateSequenceParameters:
    """
    Parameters for the GoThroughGateSequence task.
    """

    name: str = "GoThroughGateSequence"
    goal_random_position: float = 0.0
    kill_dist: float = 10.0
    gate_width: float = 1.5
    gate_thickness: float = 0.2
    num_points: int = 5

    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    contact_penalty: ContactPenalty = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_gate_position_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_gate_heading_curriculum: CurriculumParameters = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.gate_width > 0, "Gate width must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."
        assert self.num_points > 0, "Number of points must be positive."

        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.contact_penalty = ContactPenalty(**self.contact_penalty)
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
        self.spawn_gate_position_curriculum = CurriculumParameters(
            **self.spawn_gate_position_curriculum
        )
        self.spawn_gate_heading_curriculum = CurriculumParameters(
            **self.spawn_gate_heading_curriculum
        )


@dataclass
class TrackLinearVelocityParameters:
    """
    Parameters for the TrackLinearVelocity task.
    """

    name: str = "TrackLinearVelocity"
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

