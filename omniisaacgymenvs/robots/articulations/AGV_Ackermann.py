__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omni.isaac.core.robots.robot import Robot
from dataclasses import dataclass, field
from pxr import Gf, PhysxSchema
from typing import Optional
import numpy as np
import omni

from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *

from omniisaacgymenvs.robots.sensors.exteroceptive.camera_module_generator import (
    sensor_module_factory,
)

from omniisaacgymenvs.robots.articulations.utils.Types import (
    FullyFeaturedWheel,
    GeometricPrimitive,
    PhysicsMaterial,
    ActuatorCfg,
    GeometricPrimitiveFactory,
)


@dataclass
class AGVAckermannParameters:
    shape: GeometricPrimitive = field(default_factory=dict)
    front_left_wheel: FullyFeaturedWheel = field(default_factory=dict)
    front_right_wheel: FullyFeaturedWheel = field(default_factory=dict)
    rear_left_wheel: FullyFeaturedWheel = field(default_factory=dict)
    rear_right_wheel: FullyFeaturedWheel = field(default_factory=dict)
    wheel_physics_material: PhysicsMaterial = field(default_factory=dict)

    steering_dynamics: ActuatorCfg = field(default_factory=dict)
    throttle_dynamics: ActuatorCfg = field(default_factory=dict)

    mass: float = 2.5
    CoM: tuple = (0, 0, 0)
    use_software_differential: bool = True
    use_software_steering_correction: bool = True

    def __post_init__(self):
        self.shape = GeometricPrimitiveFactory.get_item(self.shape)

        self.front_left_wheel = FullyFeaturedWheel(**self.front_left_wheel)
        self.front_right_wheel = FullyFeaturedWheel(**self.front_right_wheel)
        self.rear_left_wheel = FullyFeaturedWheel(**self.rear_left_wheel)
        self.rear_right_wheel = FullyFeaturedWheel(**self.rear_right_wheel)
        self.wheel_physics_material = PhysicsMaterial(**self.wheel_physics_material)

        self.steering_dynamics = ActuatorCfg(**self.steering_dynamics)
        self.throttle_dynamics = ActuatorCfg(**self.throttle_dynamics)


class CreateAGVAckermann:
    """
    Creates a 2 wheeled Skidsteer robot."""

    def __init__(self, path: str, cfg: dict) -> None:
        self.platform_path = path
        self.joints_path = "joints"
        self.materials_path = "materials"
        self.core_path = None
        self.stage = omni.usd.get_context().get_stage()

        # Reads the thruster configuration and computes the number of virtual thrusters.
        self.settings = AGVAckermannParameters(**cfg)
        self.camera_cfg = cfg.get("camera", None)

    def build(self) -> None:
        """
        Builds the platform."""

        # Creates articulation root and the Xforms to store materials/joints.
        self.platform_path, self.platform_prim = createArticulation(
            self.stage, self.platform_path
        )
        self.joints_path, self.joints_prim = createXform(
            self.stage, self.platform_path + "/" + self.joints_path
        )
        self.materials_path, self.materials_prim = createXform(
            self.stage, self.platform_path + "/" + self.materials_path
        )

        # Creates a set of basic materials
        self.createBasicColors()

        # Creates the main body element and adds the position & heading markers.
        self.createCore()
        self.createDrivingWheels()

    def createCore(self) -> None:
        """
        Creates the core of the AMR.
        """

        self.core_path, self.core_prim = self.settings.shape.build(
            self.stage, self.platform_path + "/core"
        )
        applyMass(self.core_prim, self.settings.mass, Gf.Vec3d(0, 0, 0))
        if self.camera_cfg is not None:
            self.createCamera()
        else:
            self.settings.shape.add_orientation_marker(
                self.stage, self.core_path + "/arrow", self.colors["red"]
            )
            self.settings.shape.add_positional_marker(
                self.stage, self.core_path + "/marker", self.colors["green"]
            )

    def createDrivingWheels(self) -> None:
        """
        Creates the wheels of the AMR.
        """
        # Creates the front left wheel
        front_left_wheel_path, front_left_wheel_prim = (
            self.settings.front_left_wheel.build(
                self.stage,
                joint_path=self.joints_path + "/front_left_wheel",
                path=self.platform_path + "/front_left_wheel",
                body_path=self.core_path,
            )
        )

        # Creates the front right wheel
        front_right_wheel_path, front_right_wheel_prim = (
            self.settings.front_right_wheel.build(
                self.stage,
                joint_path=self.joints_path + "/front_right_wheel",
                path=self.platform_path + "/front_right_wheel",
                body_path=self.core_path,
            )
        )

        # Creates the rear left wheel
        rear_left_wheel_path, rear_left_wheel_prim = (
            self.settings.rear_left_wheel.build(
                self.stage,
                joint_path=self.joints_path + "/rear_left_wheel",
                path=self.platform_path + "/rear_left_wheel",
                body_path=self.core_path,
            )
        )

        # Creates the rear right wheel
        rear_right_wheel_path, rear_right_wheel_prim = (
            self.settings.rear_right_wheel.build(
                self.stage,
                joint_path=self.joints_path + "/rear_right_wheel",
                path=self.platform_path + "/rear_right_wheel",
                body_path=self.core_path,
            )
        )
        self.settings.wheel_physics_material.build(
            self.stage, self.materials_path + "/wheel_material"
        )

        mat = UsdShade.Material.Get(self.stage, self.materials_path + "/wheel_material")
        applyMaterial(front_left_wheel_prim, mat, purpose="physics")
        applyMaterial(front_right_wheel_prim, mat, purpose="physics")
        applyMaterial(rear_left_wheel_prim, mat, purpose="physics")
        applyMaterial(rear_right_wheel_prim, mat, purpose="physics")

    def createBasicColors(self) -> None:
        """
        Creates a set of basic colors."""

        self.colors = {}
        self.colors["red"] = createColor(
            self.stage, self.materials_path + "/red", [1, 0, 0]
        )
        self.colors["green"] = createColor(
            self.stage, self.materials_path + "/green", [0, 1, 0]
        )
        self.colors["blue"] = createColor(
            self.stage, self.materials_path + "/blue", [0, 0, 1]
        )
        self.colors["white"] = createColor(
            self.stage, self.materials_path + "/white", [1, 1, 1]
        )
        self.colors["grey"] = createColor(
            self.stage, self.materials_path + "/grey", [0.5, 0.5, 0.5]
        )
        self.colors["dark_grey"] = createColor(
            self.stage, self.materials_path + "/dark_grey", [0.25, 0.25, 0.25]
        )
        self.colors["black"] = createColor(
            self.stage, self.materials_path + "/black", [0, 0, 0]
        )

    def createCamera(self) -> None:
        """
        Creates a camera module prim.
        """
        self.camera = sensor_module_factory.get(self.camera_cfg["module_name"])(
            self.camera_cfg
        )
        self.camera.build()


class AGV_Ackermann(Robot):
    def __init__(
        self,
        prim_path: str,
        cfg: dict,
        name: Optional[str] = "AGV_SS_2W",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        AMR = CreateAGVAckermann(prim_path, cfg)
        AMR.build()

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )
        stage = omni.usd.get_context().get_stage()
        art = PhysxSchema.PhysxArticulationAPI.Apply(stage.GetPrimAtPath(prim_path))
        art.CreateEnabledSelfCollisionsAttr().Set(False)
