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
from typing import Optional
import numpy as np
from pxr import Gf
import torch
import omni
import carb
import math
import os

from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *
from omniisaacgymenvs.tasks.MFP3D.thruster_generator import (
    compute_actions,
)
from omniisaacgymenvs.tasks.MFP3D.thruster_generator import (
    ConfigurationParameters,
)

from omniisaacgymenvs.robots.sensors.exteroceptive.camera_module_generator import (
    sensor_module_factory,
)


@dataclass
class PlatformParameters:
    shape: str = "sphere"
    radius: float = 0.31
    height: float = 0.5
    mass: float = 5.32
    CoM: tuple = (0, 0, 0)
    refinement: int = 2
    usd_asset_path: str = "/None"
    enable_collision: bool = False

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
        assert (
            type(self.enable_collision) == bool
        ), "The enable_collision must be a bool."
        self.refinement = int(self.refinement)


class CreatePlatform:
    """
    Creates a floating platform with a core body and a set of thrusters."""

    def __init__(self, path: str, cfg: dict) -> None:
        self.platform_path = path
        self.joints_path = "joints"
        self.materials_path = "materials"
        self.core_path = None
        self.stage = omni.usd.get_context().get_stage()

        # Reads the thruster configuration and computes the number of virtual thrusters.
        self.settings = PlatformParameters(**cfg["core"])
        thruster_cfg = ConfigurationParameters(**cfg["configuration"])
        self.num_virtual_thrusters = compute_actions(thruster_cfg)
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
        if self.settings.shape == "sphere":
            self.core_path = self.createRigidSphere(
                self.platform_path + "/core",
                "body",
                self.settings.radius,
                Gf.Vec3d(*self.settings.CoM),
                self.settings.mass,
            )
        elif self.settings.shape == "cylinder":
            self.core_path = self.createRigidCylinder(
                self.platform_path + "/core",
                "body",
                self.settings.radius,
                self.settings.height,
                Gf.Vec3d(*self.settings.CoM),
                self.settings.mass,
            )
        if self.camera_cfg is not None:
            self.createCamera()
        else:
            self.createArrowXform(self.core_path + "/arrow")
            self.createPositionMarkerXform(self.core_path + "/marker")

        # Adds virtual anchors for the thrusters
        for i in range(self.num_virtual_thrusters):
            self.createVirtualThruster(
                self.platform_path + "/v_thruster_" + str(i),
                self.joints_path + "/v_thruster_joint_" + str(i),
                self.core_path,
                0.0001,
                Gf.Vec3d(0, 0, 0),
            )

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

    def createArrowXform(self, path: str) -> None:
        """
        Creates an Xform to store the arrow indicating the platform heading."""

        self.arrow_path, self.arrow_prim = createXform(self.stage, path)
        createArrow(
            self.stage,
            self.arrow_path,
            0.1,
            0.5,
            [self.settings.radius, 0, 0],
            self.settings.refinement,
        )
        applyMaterial(self.arrow_prim, self.colors["red"])

    def createPositionMarkerXform(self, path: str) -> None:
        """
        Creates an Xform to store the position marker."""

        self.marker_path, self.marker_prim = createXform(self.stage, path)
        sphere_path, sphere_geom = createSphere(
            self.stage,
            self.marker_path + "/marker_sphere_z_plus",
            0.05,
            self.settings.refinement,
        )
        setTranslate(sphere_geom, Gf.Vec3d([0, 0, self.settings.radius]))
        applyMaterial(self.stage.GetPrimAtPath(sphere_path), self.colors["blue"])
        sphere_path, sphere_geom = createSphere(
            self.stage,
            self.marker_path + "/marker_sphere_z_minus",
            0.05,
            self.settings.refinement,
        )
        setTranslate(sphere_geom, Gf.Vec3d([0, 0, -self.settings.radius]))
        applyMaterial(self.stage.GetPrimAtPath(sphere_path), self.colors["blue"])
        sphere_path, sphere_geom = createSphere(
            self.stage,
            self.marker_path + "/marker_sphere_y_plus",
            0.05,
            self.settings.refinement,
        )
        setTranslate(sphere_geom, Gf.Vec3d([0, self.settings.radius, 0]))
        applyMaterial(self.stage.GetPrimAtPath(sphere_path), self.colors["green"])
        sphere_path, sphere_geom = createSphere(
            self.stage,
            self.marker_path + "/marker_sphere_y_minus",
            0.05,
            self.settings.refinement,
        )
        setTranslate(sphere_geom, Gf.Vec3d([0, -self.settings.radius, 0]))
        applyMaterial(self.stage.GetPrimAtPath(sphere_path), self.colors["green"])
        sphere_path, sphere_geom = createSphere(
            self.stage,
            self.marker_path + "/marker_sphere_x_plus",
            0.05,
            self.settings.refinement,
        )
        setTranslate(sphere_geom, Gf.Vec3d([self.settings.radius, 0, 0]))
        applyMaterial(self.stage.GetPrimAtPath(sphere_path), self.colors["red"])
        sphere_path, sphere_geom = createSphere(
            self.stage,
            self.marker_path + "/marker_sphere_x_minus",
            0.05,
            self.settings.refinement,
        )
        setTranslate(sphere_geom, Gf.Vec3d([-self.settings.radius, 0, 0]))
        applyMaterial(self.stage.GetPrimAtPath(sphere_path), self.colors["red"])

    def createRigidSphere(
        self, path: str, name: str, radius: float, CoM: list, mass: float
    ) -> str:
        """
        Creates a rigid sphere. The sphere is a RigidBody, a Collider, and has a mass and CoM.
        It is used to create the main body of the platform."""

        # Creates an Xform to store the core body
        path, prim = createXform(self.stage, path)
        # Creates a sphere
        sphere_path = path + "/" + name
        sphere_path, sphere_geom = createSphere(
            self.stage, path + "/" + name, radius, self.settings.refinement
        )
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim, self.settings.enable_collision)
        # Sets the mass and CoM
        applyMass(sphere_prim, mass, CoM)
        return sphere_path

    def createRigidCylinder(
        self, path: str, name: str, radius: float, height: float, CoM: list, mass: float
    ) -> str:
        """
        Creates a rigid cylinder. The cylinder is a RigidBody, a Collider, and has a mass and CoM.
        It is used to create the main body of the platform."""

        # Creates an Xform to store the core body
        path, prim = createXform(self.stage, path)
        # Creates a sphere
        sphere_path = path + "/" + name
        sphere_path, sphere_geom = createCylinder(
            self.stage, path + "/" + name, radius, height, self.settings.refinement
        )
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim, self.settings.enable_collision)
        # Sets the mass and CoM
        applyMass(sphere_prim, mass, CoM)
        return sphere_path

    def createVirtualThruster(
        self, path: str, joint_path: str, parent_path: str, thruster_mass, thruster_CoM
    ) -> str:
        """
        Creates a virtual thruster. The thruster is a RigidBody, a Collider, and has a mass and CoM.
        It is used to create the thrusters of the platform."""

        # Create Xform
        thruster_path, thruster_prim = createXform(self.stage, path)
        # Add shapes
        setTranslate(thruster_prim, Gf.Vec3d([0, 0, 0]))
        setOrient(thruster_prim, Gf.Quatd(1, Gf.Vec3d([0, 0, 0])))
        # Make rigid
        applyRigidBody(thruster_prim)
        # Add mass
        applyMass(thruster_prim, thruster_mass, thruster_CoM)
        # Create joint
        createFixedJoint(self.stage, joint_path, parent_path, thruster_path)
        return thruster_path

    def createCamera(self) -> None:
        """
        Creates a camera module prim.
        """
        self.camera = sensor_module_factory.get(self.camera_cfg["module_name"])(
            self.camera_cfg
        )
        self.camera.build()


class ModularFloatingPlatform(Robot):
    def __init__(
        self,
        prim_path: str,
        cfg: dict,
        name: Optional[str] = "modular_floating_platform",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None,
    ) -> None:
        """[summary]"""

        self._usd_path = usd_path
        self._name = name

        fp = CreatePlatform(prim_path, cfg)
        fp.build()

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )
