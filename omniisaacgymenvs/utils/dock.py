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

    usd_path: str = None
    show_axis: bool = False
    mass: float = 5.0
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
        self.create_articulation_root(prim_path)
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
    
    def create_articulation_root(self, prim_path)->None:
        """
        Create a root xform and link usd to it.
        Args:
            prim_path (str): path to the prim
        """
        createArticulation(self.stage, prim_path)
        add_reference_to_stage(os.path.join(os.getcwd(), self.dock_params.usd_path), prim_path)
        axis_prim = get_prim_at_path(prim_path+"/dock/axis")
        if self.dock_params.show_axis:
            axis_prim.GetAttribute("visibility").Set("visible")
        else:
            axis_prim.GetAttribute("visibility").Set("invisible")
            
    def build(self)->None:
        """
        Apply RigidBody API, Collider, mass, and PlaneLock joints.
        """
        self.joints_path, self.joints_prim = createXform(
            self.stage, self.prim_path + "/" + self.joints_path
        )
        self.configure_core_prim()
        self.createXYPlaneLock()
    
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
    
    def configure_core_prim(self):
        """
        Configures the body of the platform.
        """
        self.core_path = self.prim_path+"/dock"
        core = get_prim_at_path(self.core_path)
        applyRigidBody(core)
        applyCollider(core, self.dock_params.enable_collision)
        applyMass(core, self.dock_params.mass)

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