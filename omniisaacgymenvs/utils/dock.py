__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import os
from typing import Optional, Sequence
from dataclasses import dataclass
from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *

from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.articulations import Articulation, ArticulationView

@dataclass
class DockParameters:
    """
    Docking station parameters.
    Args:
        usd_path (str): path to the usd file
        show_axis (bool): show the axis of the docking station
        mass (float): mass of the docking station
    """
    shape: str = "sphere"
    usd_path: str = None
    mass: float = 5.0
    radius: float = 0.31
    height: float = 0.5
    add_arrow: bool = False
    enable_collision: bool = True

class Dock(Articulation):
    """
    Class to create xform prim for a docking station.
    See parent class for more details about the arguments.
    Args:
        prim_path (str): path to the prim
        name (str): name of the prim
        position (Optional[Sequence[float]], optional): _description_. Defaults to None.
        translation (Optional[Sequence[float]], optional): _description_. Defaults to None.
        orientation (Optional[Sequence[float]], optional): _description_. Defaults to None.
        scale (Optional[Sequence[float]], optional): _description_. Defaults to None.
        visible (Optional[bool], optional): _description_. Defaults to True.
        dock_params (dict, optional): dictionary of DockParameters. Defaults to None.
    """
    def __init__(
            self,
            prim_path: str,
            name: str = "dock",
            position: Optional[Sequence[float]] = None,
            translation: Optional[Sequence[float]] = None,
            orientation: Optional[Sequence[float]] = None,
            scale: Optional[Sequence[float]] = None,
            visible: Optional[bool] = True,
            dock_params: dict = None,
            ):
        self.dock_params = DockParameters(**dock_params)
        self.stage = get_current_stage()
        self.joints_path = "joints"
        self.materials_path = "materials"
        createArticulation(self.stage, prim_path)
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
        )
        self.build()
        return
            
    def build(self)->None:
        """
        Apply RigidBody API, Collider, mass, and PlaneLock joints.
        """
        self.joints_path, self.joints_prim = createXform(
            self.stage, self.prim_path + "/" + self.joints_path
        )
        if self.dock_params.usd_path is not None:
            self.loadFromFile(self.prim_path, self.dock_params.mass)
        else:
            if self.dock_params.shape == "sphere":
                self.createRigidSphere(
                    self.prim_path, self.dock_params.radius, self.dock_params.mass
                )
            elif self.dock_params.shape == "cylinder":
                self.createRigidCylinder(
                    self.prim_path, self.dock_params.radius, self.dock_params.height, self.dock_params.mass
                )

        self.createXYPlaneLock()

        self.materials_path, self.materials_prim = createXform(
            self.stage, self.core_path + "/" + self.materials_path
        )
        # Creates a set of basic materials
        self.createBasicColors()
        if self.dock_params.add_arrow:
            self.createArrowXform(self.core_path + "/arrow")
    
    def createXYPlaneLock(self) -> None:
        """
        Creates a set of joints to constrain the platform to the XY plane.
        3DoF: translation on X and Y, rotation on Z.
        """

        # Create anchor to world. It's fixed.
        anchor_path, anchor_prim = createXform(
            self.stage, self.prim_path + "/world_anchor"
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
            self.stage, self.prim_path + "/x_translation_body"
        )
        y_tr_path, y_tr_prim = createXform(
            self.stage, self.prim_path + "/y_translation_body"
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
            self.joints_path + "/dock_world_joint_x",
            body_path1=anchor_path,
            body_path2=x_tr_path,
            axis="X",
            enable_drive=False,
        )
        tr_joint_y = createPrismaticJoint(
            self.stage,
            self.joints_path + "/dock_world_joint_y",
            body_path1=x_tr_path,
            body_path2=y_tr_path,
            axis="Y",
            enable_drive=False,
        )
        # Adds the joint allowing for rotation
        rv_joint_z = createRevoluteJoint(
            self.stage,
            self.joints_path + "/dock_world_joint_z",
            body_path1=y_tr_path,
            body_path2=self.core_path,
            axis="Z",
            enable_drive=False,
        )
    
    def createRigidSphere(
        self, path: str, radius: float, mass: float
    ) -> None:
        """
        Creates a rigid sphere. The sphere is a RigidBody, a Collider, and has a mass and CoM.
        It is used to create the main body of the platform."""

        # Creates a sphere
        sphere_path, sphere_geom = createSphere(
            self.stage, path + "/" + "dock", radius, 2
        )
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        self.core_path = sphere_path
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim, self.dock_params.enable_collision)
        # Sets the mass and CoM
        applyMass(sphere_prim, mass)

    def createRigidCylinder(
        self, path: str, radius: float, height: float, mass: float
    ) -> None:
        """
        Creates a rigid cylinder. The cylinder is a RigidBody, a Collider, and has a mass and CoM.
        It is used to create the main body of the platform."""

        # Creates a cylinder
        cylinder_path, sphere_geom = createCylinder(
            self.stage, path + "/" + "dock", radius, height, 2
        )
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        self.core_path = cylinder_path
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim, self.dock_params.enable_collision)
        # Sets the mass and CoM
        applyMass(sphere_prim, mass)
    
    def loadFromFile(self, path:str, mass:float)->None:
        """
        Load a usd file and apply RigidBody API, Collider, and mass.
        Args:
            path (str): path to the usd file
            name (str): name of the prim
            mass (float): mass of the prim
        """
        add_reference_to_stage(os.path.join(os.getcwd(), self.dock_params.usd_path), path)
        self.core_path = path+"/dock"
        core = get_prim_at_path(self.core_path)
        applyRigidBody(core)
        mesh = get_prim_at_path(self.core_path+"/mesh")
        applyCollider(mesh, self.dock_params.enable_collision)
        applyMass(core, mass)
    
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
            [self.dock_params.radius, 0, 0],
            2,
        )
        applyMaterial(self.arrow_prim, self.colors["green"])


class DockView(ArticulationView):
    def __init__(
        self, prim_paths_expr: str, 
        name: Optional[str] = "DockView", 
    ) -> None:
        """[summary]"""

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )
        self.base = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/dock/dock",
            name="dock_base_view",
        )
    
    def get_plane_lock_indices(self):
        self.lock_indices = [
            self.get_dof_index("dock_world_joint_x"),
            self.get_dof_index("dock_world_joint_y"),
            self.get_dof_index("dock_world_joint_z"),
        ]