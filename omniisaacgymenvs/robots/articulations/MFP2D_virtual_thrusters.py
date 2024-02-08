from typing import Optional

from omni.isaac.core.robots.robot import Robot

import numpy as np
import torch
import carb
import dataclasses

import omni
import math
from pxr import Gf

from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_thruster_generator import (
    compute_actions,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_core import parse_data_dict
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_thruster_generator import (
    ConfigurationParameters,
)


class CreatePlatform:
    """
    Creates a floating platform with a core body and a set of thrusters."""

    def __init__(self, path: str, cfg: dict) -> None:
        self.platform_path = path
        self.joints_path = "joints"
        self.materials_path = "materials"
        self.core_path = None
        self.stage = omni.usd.get_context().get_stage()

        self.read_cfg(cfg)

    def read_cfg(self, cfg: dict) -> None:
        """
        Reads the configuration file and sets the parameters for the platform."""

        if "core" in cfg.keys():
            if "shape" in cfg["core"].keys():
                self.core_shape = cfg["core"]["shape"]
                assert type(self.core_shape) is str
                self.core_shape.lower()
                assert (self.core_shape == "sphere") or (self.core_shape == "cylinder")
            else:
                self.core_shape = "sphere"
            if self.core_shape == "sphere":
                if "radius" in cfg["core"].keys():
                    self.core_radius = cfg["core"]["radius"]
                else:
                    self.core_radius = 0.5
            if self.core_shape == "cylinder":
                if "radius" in cfg["core"].keys():
                    self.core_radius = cfg["core"]["radius"]
                else:
                    self.core_radius = 0.5
                if "height" in cfg["core"].keys():
                    self.height_radius = cfg["core"]["radius"]
                else:
                    self.core_height = 0.5
            if "CoM" in cfg["core"].keys():
                self.core_CoM = Gf.Vec3d(list(cfg["core"]["CoM"]))
            else:
                self.core_CoM = Gf.Vec3d([0, 0, 0])
            if "mass" in cfg["core"].keys():
                self.core_mass = cfg["core"]["mass"]
            else:
                self.core_mass = 5.0
            if "refinement" in cfg.keys():
                self.refinement = cfg["refinement"]
            else:
                self.refinement = 2
        else:
            self.core_shape = "sphere"
            self.core_radius = 0.5
            self.core_CoM = Gf.Vec3d([0, 0, 0])
            self.core_mass = 5.0
            self.refinement = 2
        # Reads the thruster configuration and computes the number of virtual thrusters.
        thruster_cfg = parse_data_dict(ConfigurationParameters(), cfg["configuration"])
        self.num_virtual_thrusters = compute_actions(thruster_cfg)

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
        if self.core_shape == "sphere":
            self.core_path = self.createRigidSphere(
                self.platform_path + "/core",
                "body",
                self.core_radius,
                self.core_CoM,
                self.core_mass,
            )
        elif self.core_shape == "cylinder":
            self.core_path = self.createRigidCylinder(
                self.platform_path + "/core",
                "body",
                self.core_radius,
                self.core_height,
                self.core_CoM,
                self.core_mass,
            )

        # Creates a set of joints to constrain the platform on the XY plane (3DoF).
        self.createXYPlaneLock()

        # Visualization
        self.createArrowXform(self.core_path + "/arrow")
        self.createPositionMarkerXform(self.core_path + "/marker")

        for i in range(self.num_virtual_thrusters):
            self.createVirtualThruster(
                self.platform_path + "/v_thruster_" + str(i),
                self.joints_path + "/v_thruster_joint_" + str(i),
                self.core_path,
                0.0001,
                Gf.Vec3d([0, 0, 0]),
            )

    def createXYPlaneLock(self) -> None:
        """
        Creates a set of joints to constrain the platform to the XY plane.
        3DoF: translation on X and Y, rotation on Z."""

        # Create anchor to world. It's fixed.
        anchor_path, anchor_prim = createXform(
            self.stage, self.platform_path + "/world_anchor"
        )
        setTranslate(anchor_prim, Gf.Vec3d(0, 0, 0))
        setOrient(anchor_prim, Gf.Quatd(1, Gf.Vec3d(0, 0, 0)))
        applyRigidBody(anchor_prim)
        applyMass(anchor_prim, 0.0000001)
        fixed_joint = createFixedJoint(
            self.stage, self.joints_path, body_path2=anchor_path
        )
        # Create the bodies & joints allowing translation
        x_tr_path, x_tr_prim = createXform(
            self.stage, self.platform_path + "/x_translation_body"
        )
        y_tr_path, y_tr_prim = createXform(
            self.stage, self.platform_path + "/y_translation_body"
        )
        setTranslate(x_tr_prim, Gf.Vec3d(0, 0, 0))
        setOrient(x_tr_prim, Gf.Quatd(1, Gf.Vec3d(0, 0, 0)))
        applyRigidBody(x_tr_prim)
        applyMass(x_tr_prim, 0.0000001)
        setTranslate(y_tr_prim, Gf.Vec3d(0, 0, 0))
        setOrient(y_tr_prim, Gf.Quatd(1, Gf.Vec3d(0, 0, 0)))
        applyRigidBody(y_tr_prim)
        applyMass(y_tr_prim, 0.0000001)
        tr_joint_x = createPrismaticJoint(
            self.stage,
            self.joints_path + "/fp_world_joint_x",
            body_path1=anchor_path,
            body_path2=x_tr_path,
            axis="X",
            enable_drive=False,
        )
        tr_joint_y = createPrismaticJoint(
            self.stage,
            self.joints_path + "/fp_world_joint_y",
            body_path1=x_tr_path,
            body_path2=y_tr_path,
            axis="Y",
            enable_drive=False,
        )
        # Adds the joint allowing for rotation
        rv_joint_z = createRevoluteJoint(
            self.stage,
            self.joints_path + "/fp_world_joint_z",
            body_path1=y_tr_path,
            body_path2=self.core_path,
            axis="Z",
            enable_drive=False,
        )

        self.tr_x_axis = "fp_world_joint_x"
        self.tr_y_axis = "fp_world_joint_y"
        self.rv_z_axis = "fp_world_joint_z"

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
            [self.core_radius, 0, 0],
            self.refinement,
        )
        applyMaterial(self.arrow_prim, self.colors["blue"])

    def createPositionMarkerXform(self, path: str) -> None:
        """
        Creates an Xform to store the position marker."""

        self.marker_path, self.marker_prim = createXform(self.stage, path)
        sphere_path, sphere_geom = createSphere(
            self.stage, self.marker_path + "/marker_sphere", 0.05, self.refinement
        )
        setTranslate(sphere_geom, Gf.Vec3d([0, 0, self.core_radius]))
        applyMaterial(self.marker_prim, self.colors["green"])

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
            self.stage, path + "/" + name, radius, self.refinement
        )
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim)
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
            self.stage, path + "/" + name, radius, height, self.refinement
        )
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim)
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

        self.joints = {
            "x_tr_axis": fp.tr_x_axis,
            "y_tr_axis": fp.tr_y_axis,
            "z_rv_axis": fp.rv_z_axis,
        }
