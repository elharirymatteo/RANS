__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


from typing import Optional, Sequence
import numpy as np
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omniisaacgymenvs.utils.shape_utils import Pin3D


class VisualPin3D(XFormPrim, Pin3D):
    """_summary_

    Args:
        prim_path (str): _description_
        name (str, optional): _description_. Defaults to "visual_arrow".
        position (Optional[Sequence[float]], optional): _description_. Defaults to None.
        translation (Optional[Sequence[float]], optional): _description_. Defaults to None.
        orientation (Optional[Sequence[float]], optional): _description_. Defaults to None.
        scale (Optional[Sequence[float]], optional): _description_. Defaults to None.
        visible (Optional[bool], optional): _description_. Defaults to True.
        color (Optional[np.ndarray], optional): _description_. Defaults to None.
        radius (Optional[float], optional): _description_. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.

    Raises:
        Exception: _description_
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
        color: Optional[np.ndarray] = None,
        ball_radius: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
    ) -> None:
        if visible is None:
            visible = True
        XFormPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
        )
        Pin3D.__init__(self, prim_path, ball_radius, poll_radius, poll_length)
        self.setBallRadius(ball_radius)
        self.setPollRadius(poll_radius)
        self.setPollLength(poll_length)
        return


class FixedPin3D(VisualPin3D):
    """_summary_

    Args:
        prim_path (str): _description_
        name (str, optional): _description_. Defaults to "fixed_sphere".
        position (Optional[np.ndarray], optional): _description_. Defaults to None.
        translation (Optional[np.ndarray], optional): _description_. Defaults to None.
        orientation (Optional[np.ndarray], optional): _description_. Defaults to None.
        scale (Optional[np.ndarray], optional): _description_. Defaults to None.
        visible (Optional[bool], optional): _description_. Defaults to None.
        color (Optional[np.ndarray], optional): _description_. Defaults to None.
        radius (Optional[np.ndarray], optional): _description_. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.
        physics_material (Optional[PhysicsMaterial], optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "fixed_arrow",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        ball_radius: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
    ) -> None:
        if not is_prim_path_valid(prim_path):
            # set default values if no physics material given
            if physics_material is None:
                static_friction = 0.2
                dynamic_friction = 1.0
                restitution = 0.0
                physics_material_path = find_unique_string_name(
                    initial_name="/World/Physics_Materials/physics_material",
                    is_unique_fn=lambda x: not is_prim_path_valid(x),
                )
                physics_material = PhysicsMaterial(
                    prim_path=physics_material_path,
                    dynamic_friction=dynamic_friction,
                    static_friction=static_friction,
                    restitution=restitution,
                )
        VisualPin3D.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            visual_material=visual_material,
        )
        # XFormPrim.set_collision_enabled(self, True)
        # if physics_material is not None:
        #    FixedArrow.apply_physics_material(self, physics_material)
        return


class DynamicPin3D(RigidPrim, FixedPin3D):
    """_summary_

    Args:
        prim_path (str): _description_
        name (str, optional): _description_. Defaults to "dynamic_sphere".
        position (Optional[np.ndarray], optional): _description_. Defaults to None.
        translation (Optional[np.ndarray], optional): _description_. Defaults to None.
        orientation (Optional[np.ndarray], optional): _description_. Defaults to None.
        scale (Optional[np.ndarray], optional): _description_. Defaults to None.
        visible (Optional[bool], optional): _description_. Defaults to None.
        color (Optional[np.ndarray], optional): _description_. Defaults to None.
        radius (Optional[np.ndarray], optional): _description_. Defaults to None.
        visual_material (Optional[VisualMaterial], optional): _description_. Defaults to None.
        physics_material (Optional[PhysicsMaterial], optional): _description_. Defaults to None.
        mass (Optional[float], optional): _description_. Defaults to None.
        density (Optional[float], optional): _description_. Defaults to None.
        linear_velocity (Optional[Sequence[float]], optional): _description_. Defaults to None.
        angular_velocity (Optional[Sequence[float]], optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "dynamic_sphere",
        position: Optional[np.ndarray] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
        visible: Optional[bool] = None,
        color: Optional[np.ndarray] = None,
        ball_radius: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
        physics_material: Optional[PhysicsMaterial] = None,
        mass: Optional[float] = None,
        density: Optional[float] = None,
        linear_velocity: Optional[Sequence[float]] = None,
        angular_velocity: Optional[Sequence[float]] = None,
    ) -> None:
        if not is_prim_path_valid(prim_path):
            if mass is None:
                mass = 0.02
        FixedPin3D.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            ball_radius=ball_radius,
            poll_radius=poll_radius,
            poll_length=poll_length,
            visual_material=visual_material,
            physics_material=physics_material,
        )
        RigidPrim.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            mass=mass,
            density=density,
            linear_velocity=linear_velocity,
            angular_velocity=angular_velocity,
        )
