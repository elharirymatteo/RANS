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


@dataclass
class MassDistributionDisturbanceParameters:
    """
    This class provides an interface to adjust the hyperparameters of the mass distribution disturbance.
    """

    mass_curriculum: CurriculumParameters = field(default_factory=dict)
    com_curriculum: CurriculumParameters = field(default_factory=dict)
    mi_curriculum: CurriculumParameters = field(default_factory=dict)

    enable: bool = False

    def __post_init__(self):
        self.mass_curriculum = CurriculumParameters(**self.mass_curriculum)
        self.com_curriculum = CurriculumParameters(**self.com_curriculum)
        self.mi_curriculum = CurriculumParameters(**self.mi_curriculum)


@dataclass
class ForceDisturbanceParameters:
    """
    This class provides an interface to adjust the hyperparameters of the force disturbance.
    """

    force_curriculum: CurriculumParameters = field(default_factory=dict)
    use_sinusoidal_patterns: bool = False
    min_freq: float = 0.1
    max_freq: float = 5.0
    min_offset: float = 0.0
    max_offset: float = 1.0
    enable: bool = False

    def __post_init__(self):
        self.force_curriculum = CurriculumParameters(**self.force_curriculum)
        assert self.min_freq > 0, "The minimum frequency must be positive."
        assert self.max_freq > 0, "The maximum frequency must be positive."
        assert (
            self.max_freq > self.min_freq
        ), "The maximum frequency must be larger than the minimum frequency."


@dataclass
class TorqueDisturbanceParameters:
    """
    This class provides an interface to adjust the hyperparameters of the force disturbance.
    """

    torque_curriculum: CurriculumParameters = field(default_factory=dict)
    use_sinusoidal_patterns: bool = False
    min_freq: float = 0.1
    max_freq: float = 5.0
    min_offset: float = 0.0
    max_offset: float = 1.0
    enable: bool = False

    def __post_init__(self):
        self.torque_curriculum = CurriculumParameters(**self.torque_curriculum)
        assert self.min_freq > 0, "The minimum frequency must be positive."
        assert self.max_freq > 0, "The maximum frequency must be positive."
        assert (
            self.max_freq > self.min_freq
        ), "The maximum frequency must be larger than the minimum frequency."


@dataclass
class NoisyObservationsParameters:
    """
    This class provides an interface to adjust the hyperparameters of the observation noise.
    """

    position_curriculum: CurriculumParameters = field(default_factory=dict)
    velocity_curriculum: CurriculumParameters = field(default_factory=dict)
    orientation_curriculum: CurriculumParameters = field(default_factory=dict)
    enable_position_noise: bool = False
    enable_velocity_noise: bool = False
    enable_orientation_noise: bool = False

    def __post_init__(self):
        self.position_curriculum = CurriculumParameters(**self.position_curriculum)
        self.velocity_curriculum = CurriculumParameters(**self.velocity_curriculum)
        self.orientation_curriculum = CurriculumParameters(
            **self.orientation_curriculum
        )


@dataclass
class NoisyActionsParameters:
    """
    This class provides an interface to adjust the hyperparameters of the action noise.
    """

    action_curriculum: CurriculumParameters = field(default_factory=dict)
    enable: bool = False

    def __post_init__(self):
        self.action_curriculum = CurriculumParameters(**self.action_curriculum)

@dataclass
class NoisyImagesParameters:
    """
    This class provides an interface to adjust the hyperparameters of the action noise.
    """

    image_curriculum: CurriculumParameters = field(default_factory=dict)
    enable: bool = False
    modality: bool = "rgb"

    def __post_init__(self):
        self.image_curriculum = CurriculumParameters(**self.image_curriculum)


@dataclass
class DisturbancesParameters:
    """
    Collection of disturbances.
    """

    mass_disturbance: MassDistributionDisturbanceParameters = field(
        default_factory=dict
    )
    force_disturbance: ForceDisturbanceParameters = field(default_factory=dict)
    torque_disturbance: TorqueDisturbanceParameters = field(default_factory=dict)
    observations_disturbance: NoisyObservationsParameters = field(default_factory=dict)
    actions_disturbance: NoisyActionsParameters = field(default_factory=dict)
    rgb_disturbance: NoisyImagesParameters = field(default_factory=dict)
    depth_disturbance: NoisyImagesParameters = field(default_factory=dict)

    def __post_init__(self):
        self.mass_disturbance = MassDistributionDisturbanceParameters(
            **self.mass_disturbance
        )
        self.force_disturbance = ForceDisturbanceParameters(**self.force_disturbance)
        self.torque_disturbance = TorqueDisturbanceParameters(**self.torque_disturbance)
        self.observations_disturbance = NoisyObservationsParameters(
            **self.observations_disturbance
        )
        self.actions_disturbance = NoisyActionsParameters(**self.actions_disturbance)
        self.rgb_disturbance = NoisyImagesParameters(**self.rgb_disturbance)
        self.depth_disturbance = NoisyImagesParameters(**self.depth_disturbance)