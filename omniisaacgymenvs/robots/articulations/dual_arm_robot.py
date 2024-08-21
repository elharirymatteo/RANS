__author__ = "Matteo El Hariry, Antoine Richard"
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
from omniisaacgymenvs.robots import actuators
from pxr import Gf, PhysxSchema, UsdShade
from typing import Optional
import numpy as np
import omni

from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *

from omniisaacgymenvs.robots.sensors.exteroceptive.camera_module_generator import (
    sensor_module_factory,
)

from omniisaacgymenvs.robots.articulations.utils.Types import (
    GeometricPrimitive,
    ActuatorCfg,
    GeometricPrimitiveFactory,
    PassiveWheelFactory,
    RigidBody,
)


@dataclass
class DualArmRobotParameters:
    base: RigidBody = field(default_factory=dict)
    links: list = field(default_factory=list)
    end_effectors: list = field(default_factory=list)
    actuators: ActuatorCfg = field(default_factory=dict)

    def __post_init__(self):
        self.base = RigidBody(**self.base)
        self.links = [RigidBody(**link) for link in self.links]
        self.end_effectors = [RigidBody(**ee) for ee in self.end_effectors]
        self.actuators = ActuatorCfg(**self.actuators)


class CreateDualArmRobot:
    def __init__(self, path: str, cfg: dict) -> None:
        self.platform_path = path
        self.joints_path = "joints"
        self.materials_path = "materials"
        self.core_path = None
        self.stage = omni.usd.get_context().get_stage()

        self.settings = DualArmRobotParameters(**cfg)
        self.camera_cfg = cfg.get("camera", None)

    def build(self) -> None:
        self.platform_path, self.platform_prim = createArticulation(
            self.stage, self.platform_path
        )
        self.joints_path, self.joints_prim = createXform(
            self.stage, self.platform_path + "/" + self.joints_path
        )
        self.materials_path, self.materials_prim = createXform(
            self.stage, self.platform_path + "/" + self.materials_path
        )

        self.createBasicColors()
        self.createCore()
        self.createLinks()
        self.createEndEffectors()

    def createCore(self) -> None:
        self.core_path, self.core_prim = self.settings.base.shape.build(
            self.stage, self.platform_path + "/base"
        )
        applyMass(self.core_prim, self.settings.base.mass, Gf.Vec3d(*self.settings.base.CoM))
        if self.camera_cfg is not None:
            self.createCamera()
        else:
            self.settings.base.shape.add_orientation_marker(
                self.stage, self.core_path + "/arrow", self.colors["red"]
            )
            self.settings.base.shape.add_positional_marker(
                self.stage, self.core_path + "/marker", self.colors["green"]
            )

    def createLinks(self) -> None:
        for i, link in enumerate(self.settings.links):
            _, _, _, link_prim = link.build(
                self.stage,
                path=self.joints_path + f"/link_{i+1}",
                #wheel_path=self.platform_path + f"/link_{i+1}",
                #body_path=self.core_path,
            )

    def createEndEffectors(self) -> None:
        for i, ee in enumerate(self.settings.end_effectors):
            _, _, _, ee_prim = ee.build(
                self.stage,
                path=self.joints_path + f"/end_effector_{i+1}",
                #path=self.platform_path + f"/end_effector_{i+1}",
                #body_path=self.core_path,
            )

    def createBasicColors(self) -> None:
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
        self.camera = sensor_module_factory.get(self.camera_cfg["module_name"])(
            self.camera_cfg
        )
        self.camera.build()


class DualArmRobot(Robot):
    def __init__(
        self,
        prim_path: str,
        cfg: dict,
        name: Optional[str] = "DualArmRobot",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None,
    ) -> None:
        self._usd_path = usd_path
        self._name = name

        robot = CreateDualArmRobot(prim_path, cfg)
        robot.build()
        self._settings = robot.settings

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
