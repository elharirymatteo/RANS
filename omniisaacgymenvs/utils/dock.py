from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.prims import XFormPrim
from typing import Optional, Sequence
from omniisaacgymenvs.robots.articulations.utils.MFP_utils import applyCollider, applyRigidBody

class Dock(XFormPrim):
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
            usd_path:str = None):
        super().__init__(
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
        )
        self.usd_path = usd_path
        self.build()
        return
    
    def build(self)->None:
        """
        Load a dock station prim and apply physics."""
        add_reference_to_stage(self.usd_path, self.prim_path)
        applyRigidBody(self.prim)