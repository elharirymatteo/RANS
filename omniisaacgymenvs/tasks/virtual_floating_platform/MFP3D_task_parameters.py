__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


@dataclass
class GoToXYZParameters:
    """
    Parameters for the GoToXY task."""

    position_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_position: float = 0.0
    max_spawn_dist: float = 6.0
    min_spawn_dist: float = 3.0
    kill_dist: float = 8.0
    boundary_cost: float = 25

    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 0.5
    spawn_curriculum_max_dist: float = 2.5
    spawn_curriculum_kill_dist: float = 3.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 750

    def __post_init__(self) -> None:
        """
        Checks that the curicullum parameters are valid."""

        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_mode = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0


@dataclass
class GoToPoseParameters:
    """
    Parameters for the GoToPose task."""

    position_tolerance: float = 0.01
    orientation_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_position: float = 0.0
    max_spawn_dist: float = 6.0
    min_spawn_dist: float = 3.0
    kill_dist: float = 8.0

    spawn_curriculum: bool = False
    spawn_curriculum_min_dist: float = 0.5
    spawn_curriculum_max_dist: float = 2.5
    spawn_curriculum_kill_dist: float = 3.0
    spawn_curriculum_mode: str = "linear"
    spawn_curriculum_warmup: int = 250
    spawn_curriculum_end: int = 750

    def __post_init__(self) -> None:
        """
        Checks that the curicullum parameters are valid."""

        assert self.spawn_curriculum_mode.lower() in [
            "linear"
        ], "Linear is the only currently supported mode."
        if not self.spawn_curriculum:
            self.spawn_curriculum_max_dist = 0
            self.spawn_curriculum_min_dist = 0
            self.spawn_curriculum_kill_dist = 0
            self.spawn_curriculum_warmup = 0
            self.spawn_curriculum_end = 0


@dataclass
class TrackXYVelocityParameters:
    """
    Parameters for the TrackXYVelocity task."""

    lin_vel_tolerance: float = 0.01
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_velocity: float = 0.75
    kill_dist: float = 500.0


@dataclass
class TrackXYOVelocityParameters:
    """
    Parameters for the TrackXYOVelocity task."""

    lin_vel_tolerance: float = 0.01
    ang_vel_tolerance: float = 0.025
    kill_after_n_steps_in_tolerance: int = 50
    goal_random_linear_velocity: float = 0.75
    goal_random_angular_velocity: float = 1
    kill_dist: float = 500.0
