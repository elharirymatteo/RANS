from dataclasses import dataclass, field
from typing import List
import math


@dataclass
class CoreParameters:
    shape: str = "sphere"
    radius: float = 0.31
    height: float = 0.5
    mass: float = 5.32
    CoM: tuple = (0, 0, 0)
    refinement: int = 2
    usd_asset_path: str = "/None"

    def __post_init__(self):
        assert self.shape in [
            "cylinder",
            "sphere",
            "asset",
        ], "The shape must be 'cylinder', 'sphere' or 'asset'."
        assert self.radius > 0, "The radius must be larger than 0."
        assert self.height > 0, "The height must be larger than 0."
        assert self.mass > 0, "The mass must be larger than 0."
        assert len(self.CoM) == 3, "The length of the CoM coordinates must be 3."
        assert self.refinement > 0, "The refinement level must be larger than 0."
        self.refinement = int(self.refinement)


@dataclass
class ThrusterParameters:
    """
    The definition of a basic thruster.
    """

    max_force: float = 1.0
    position: tuple = (0, 0, 0)
    orientation: tuple = (0, 0, 0)
    delay: float = 0.0
    response_order: int = 0
    tau: float = 1.0

    def __post_init__(self):
        assert self.tau > 0, "The response time of the system must be larger than 0"
        assert self.response_order in [
            0,
            1,
        ], "The response order of the system must be 0 or 1."
        assert (
            self.delay >= 0
        ), "The delay in system response must larger or equal to 0."


@dataclass
class ReactionWheelParameters:
    """
    The definition of a basic reaction wheel.
    """

    mass: float = 0.250
    inertia: float = 0.3
    position: tuple = (0, 0, 0)
    orientation: tuple = (0, 0, 0)
    max_speed: float = 5000
    delay: float = 0.0
    response_order: float = 1
    tau: float = 1.0

    def __post_init__(self):
        assert self.tau > 0, "The response time of the system must be larger than 0"
        assert self.response_order in [
            0,
            1,
        ], "The response order of the system must be 0 or 1."
        assert (
            self.delay >= 0
        ), "The delay in system response must larger or equal to 0."
        assert (
            self.max_speed > 0
        ), "The maximum speed of the reaction wheel must be larger than 0."


@dataclass
class FloatingPlatformParameters:
    """
    Thruster configuration parameters.
    """

    use_four_configurations: bool = False
    num_anchors: int = 4
    offset: float = math.pi / 4
    thrust_force: float = 1.0
    visualize: bool = False
    save_path: str = "thruster_configuration.png"
    thruster_model: ThrusterParameters = field(default_factory=dict)
    reaction_wheel_model: ReactionWheelParameters = field(default_factory=dict)

    def __post_init__(self):
        assert self.num_anchors > 1, "num_anchors must be larger or equal to 2."

    def generate_anchors_2D(self, radius):
        for i in range(self.num_anchors):
            math.pi * 2 * i / self.num_anchors
        pass

    def generate_anchors_3D(self, radius):
        pass


@dataclass
class SpaceCraftDefinition:
    """
    The definition of the spacecraft / floating platform.
    """

    use_floating_platform_generation = True
    core: CoreParameters = field(default_factory=dict)
    floating_platform: FloatingPlatformParameters = field(default_factory=dict)
    thrusters: List[ThrusterParameters] = field(default_factory=list)
    reaction_wheels: List[ReactionWheelParameters] = field(default_factory=list)

    def __post_init__(self):
        self.core = CoreParameters(**self.core)
        if self.use_floating_platform_generation == False:
            raise NotImplementedError


@dataclass
class PlatformRandomization:
    """
    Platform randomization parameters.
    """

    random_permutation: bool = False
    random_offset: bool = False
    randomize_thruster_position: bool = False
    min_random_radius: float = 0.125
    max_random_radius: float = 0.25
    random_theta: float = 0.125
    randomize_thrust_force: bool = False
    min_thrust_force: float = 0.5
    max_thrust_force: float = 1.0
    kill_thrusters: bool = False
    max_thruster_kill: int = 1


def compute_actions(cfg_param: FloatingPlatformParameters):
    """
    Computes the number of actions for the thruster configuration.
    """

    if cfg_param.use_four_configurations:
        return 10
    else:
        return cfg_param.num_anchors * 4
