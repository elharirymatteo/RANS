from omni.isaac.core.robots.robot import Robot

from typing import Optional, List, Tuple
import dataclasses
import numpy as np
import torch
import carb
import os

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
        """
        Creates a floating platform with a core body and a set of thrusters.

        Args:
            path (str): The path to the platform.
            cfg (dict): The configuration file.
        """

        self.platform_path = path
        self.joints_path = "joints"
        self.materials_path = "materials"
        self.core_path = None
        self.stage = omni.usd.get_context().get_stage()

        self.read_cfg(cfg)

    def read_cfg(self, cfg: dict) -> None:
        """
        Reads the configuration file and sets the parameters for the platform.

        Args:
            cfg (dict): The configuration file.
        """

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
        Builds the platform.
        """

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
                Gf.Vec3d(0, 0, 0),
                1e-4,
            )
        elif self.core_shape == "cylinder":
            self.core_path = self.createRigidCylinder(
                self.platform_path + "/core",
                "body",
                self.core_radius,
                self.core_height,
                Gf.Vec3d(0, 0, 0),
                1e-4,
            )
        self.createMovableCoM(
            self.platform_path + "/movable_CoM",
            "CoM",
            self.core_radius / 2,
            self.core_CoM,
            self.core_mass,
        )
        self.createArrowXform(self.core_path + "/arrow")
        self.createPositionMarkerXform(self.core_path + "/marker")

        # Adds virtual anchors for the thrusters
        for i in range(self.num_virtual_thrusters):
            self.createVirtualThruster(
                self.platform_path + "/v_thruster_" + str(i),
                self.joints_path + "/v_thruster_joint_" + str(i),
                self.core_path,
                0.0001,
                Gf.Vec3d([0, 0, 0]),
            )

    def createMovableCoM(
        self, path: str, name: str, radius: float, CoM: Gf.Vec3d, mass: float
    ) -> None:
        """
        Creates a movable Center of Mass (CoM).

        Args:
            path (str): The path to the movable CoM.
            name (str): The name of the sphere used as CoM.
            radius (float): The radius of the sphere used as CoM.
            CoM (Gf.Vec3d): The resting position of the center of mass.
            mass (float): The mass of the Floating Platform.
        """

        # Create Xform
        CoM_path, CoM_prim = createXform(self.stage, path)
        # Add shapes
        sphere_path = CoM_path + "/" + name
        sphere_path, sphere_geom = createSphere(
            self.stage, CoM_path + "/" + name, radius, self.refinement
        )
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim)
        # Sets the mass and CoM
        applyMass(sphere_prim, mass, Gf.Vec3d(0, 0, 0))

        # Add dual prismatic joint
        CoM_path, CoM_prim = createXform(self.stage, "CoM_joints")
        createP2Joint(
            self.stage, self.joints_path + "/CoM_joints", self.core_path, sphere_path
        )

        self.CoM_x_axis = os.path.join(
            path, self.joints_path, "CoM_joints", "x_axis_joint"
        )
        self.CoM_y_axis = os.path.join(
            path, self.joints_path, "CoM_joints", "y_axis_joint"
        )
        return sphere_path

    def createBasicColors(self) -> None:
        """
        Creates a set of basic colors.
        """

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
        Creates an Xform to store the arrow indicating the platform heading.

        Args:
            path (str): The path to the arrow.
        """

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
        It is used to create the main body of the platform.

        Args:
            path (str): The path to the sphere.
            name (str): The name of the sphere.
            radius (float): The radius of the sphere.
            CoM (list): The center of mass of the sphere.
            mass (float): The mass of the sphere.

        Returns:
            str: The path to the sphere.
        """

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
        It is used to create the main body of the platform.

        Args:
            path (str): The path to the cylinder.
            name (str): The name of the cylinder.
            radius (float): The radius of the cylinder.
            height (float): The height of the cylinder.
            CoM (list): The center of mass of the cylinder.
            mass (float): The mass of the cylinder.

        Returns:
            str: The path to the cylinder.
        """

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
        It is used to create the thrusters of the platform.

        Args:
            path (str): The path to the thruster.
            joint_path (str): The path to the joint.
            parent_path (str): The path to the parent.
            thruster_mass (float): The mass of the thruster.
            thruster_CoM (list): The center of mass of the thruster.

        Returns:
            str: The path to the thruster.
        """

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
        """
        Creates a modular floating platform.

        Args:
            prim_path (str): The path to the platform.
            cfg (dict): The configuration file.
            name (str, optional): The name of the platform. Defaults to 'modular_floating_platform'.
            usd_path (str, optional): The path to the USD file. Defaults to None.
            translation (np.ndarray, optional): The translation of the platform. Defaults to None.
            orientation (np.ndarray, optional): The orientation of the platform. Defaults to None.
            scale (np.array, optional): The scale of the platform. Defaults to None.
        """

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
