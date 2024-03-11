import os
from typing import Optional, Sequence
from dataclasses import dataclass
from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *

from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import RigidPrim

@dataclass
class DockParameters:
    """
    Platform physical parameters."""

    usd_path: str = None
    show_axis: bool = False
    mass: float = 5.0

class Dock(RigidPrim):
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
        usd_path (str): path to the reference usd file.
    """
    def __init__(
            self,
            prim_path: str,
            name: str = "visual_pin",
            position: Optional[Sequence[float]] = None,
            translation: Optional[Sequence[float]] = None,
            orientation: Optional[Sequence[float]] = None,
            scale: Optional[Sequence[float]] = None,
            visible: Optional[bool] = True,
            dock_params: dict = None,
            ):
        self.dock_params = DockParameters(**dock_params)
        self.stage = get_current_stage()
        self.create_root_prim(prim_path)
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
    
    def create_root_prim(self, prim_path)->None:
        """
        Create a root xform and link usd to it."""
        createXform(self.stage, prim_path)
        add_reference_to_stage(os.path.join(os.getcwd(), self.dock_params.usd_path), prim_path)
        axis_prim = get_prim_at_path(prim_path+"/dock/axis")
        if self.dock_params.show_axis:
            axis_prim.GetAttribute("visibility").Set("visible")
        else:
            axis_prim.GetAttribute("visibility").Set("invisible")

    def build(self)->None:
        """
        Apply RigidBody API, Collider, mass, and D6 joint."""
        applyRigidBody(self.prim)
        applyCollider(self.prim, True)
        applyMass(self.prim, self.dock_params.mass)
        create3DOFJoint(self.stage, self.prim_path+"/d6joint", "/World/envs/env_0", self.prim_path)