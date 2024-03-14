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
    ConeShapePenalty,
)

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class GoToXYParameters:
    """
    Parameters for the GoToXY task.
    """

    name: str = "GoToXY"
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
class TrackXYVelocityParameters:
    """
    Parameters for the TrackXYVelocity task.
    """

    name: str = "TrackXYVelocity"
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
class TrackXYOVelocityParameters:
    """
    Parameters for the TrackXYOVelocity task.
    """

    name: str = "TrackXYOVelocity"
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


@dataclass
class TrackXYVelocityHeadingParameters:
    """
    Parameters for the TrackXYVelocityHeading task.
    """

    name: str = "TrackXYVelocityHeading"
    velocity_tolerance: float = 0.01
    heading_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_position: float = 0.0
    kill_dist: float = 500.0

    target_linear_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        assert self.velocity_tolerance > 0, "Velocity tolerance must be positive."
        assert self.heading_tolerance > 0, "Heading tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.goal_random_position >= 0, "Goal random position must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."

        self.target_linear_velocity_curriculum = CurriculumParameters(
            **self.target_linear_velocity_curriculum
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
class CloseProximityDockParameters:
    """
    Parameters for the GoToPose task."""
    name: str = "CloseProximityDock"
    position_tolerance: float = 0.01
    heading_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 50
    kill_dist: float = 10.0
    collision_force_tolerance: float = 0.25
    dock_footprint_diameter: float = 0.8
    fp_footprint_diameter: float = 0.7
    goal_to_penalty_anchor_dist: float = 0.4
    env_x: float = 3.0
    env_y: float = 5.0
    kill_relative_angle: float = 1.571 #pi/2
    
    boundary_penalty: BoundaryPenalty = field(default_factory=dict)
    relative_angle_penalty: ConeShapePenalty = field(default_factory=dict)
    
    spawn_dock_mass_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_dock_space_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_position_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_relative_angle_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_heading_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_linear_velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    spawn_angular_velocity_curriculum: CurriculumParameters = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.position_tolerance > 0, "Position tolerance must be positive."
        assert self.heading_tolerance > 0, "Heading tolerance must be positive."
        assert (
            self.kill_after_n_steps_in_tolerance > 0
        ), "Kill after n steps in tolerance must be positive."
        assert self.kill_dist > 0, "Kill distance must be positive."
        assert self.collision_force_tolerance > 0, "Collision force tolerance must be positive."
        assert self.dock_footprint_diameter > 0, "Dock footprint diameter must be positive."
        assert self.fp_footprint_diameter > 0, "FP footprint diameter must be positive."
        assert self.goal_to_penalty_anchor_dist > 0, "Goal to penalty anchor distance must be positive."
        assert self.env_x > 0, "Environment x dimension must be positive."
        assert self.env_y > 0, "Environment y dimension must be positive."
        assert self.kill_relative_angle > 0, "Kill relative angle must be positive."
        
        self.boundary_penalty = BoundaryPenalty(**self.boundary_penalty)
        self.relative_angle_penalty = ConeShapePenalty(**self.relative_angle_penalty)
        
        self.spawn_dock_mass_curriculum = CurriculumParameters(
            **self.spawn_dock_mass_curriculum
        )
        
        self.spawn_dock_space_curriculum = CurriculumParameters(
            **self.spawn_dock_space_curriculum
        )
        
        self.spawn_position_curriculum = CurriculumParameters(
            **self.spawn_position_curriculum
        )

        self.spawn_relative_angle_curriculum = CurriculumParameters(
            **self.spawn_relative_angle_curriculum
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