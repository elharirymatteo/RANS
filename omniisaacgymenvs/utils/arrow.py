# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#
from typing import Optional, Sequence
import numpy as np
from omni.isaac.core.materials.visual_material import VisualMaterial
from omni.isaac.core.prims.rigid_prim import RigidPrim
from omni.isaac.core.prims.xform_prim import XFormPrim
from omni.isaac.core.prims.geometry_prim import GeometryPrim
from omni.isaac.core.materials import PreviewSurface
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.utils.string import find_unique_string_name
from pxr import UsdGeom, Gf
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.stage import get_current_stage

def setXformOp(prim, value, property):
    xform = UsdGeom.Xformable(prim)
    op = None
    for xformOp in xform.GetOrderedXformOps():
        if xformOp.GetOpType() == property:
            op = xformOp
    if op:
        xform_op = op
    else:
        xform_op = xform.AddXformOp(property, UsdGeom.XformOp.PrecisionDouble, "")
    xform_op.Set(value)

def setScale(prim, value):
    setXformOp(prim, value, UsdGeom.XformOp.TypeScale)

def setTranslate(prim, value):
    setXformOp(prim, value, UsdGeom.XformOp.TypeTranslate)

def setRotateXYZ(prim, value):
    setXformOp(prim, value, UsdGeom.XformOp.TypeRotateXYZ)
    
def setOrient(prim, value):
    setXformOp(prim, value, UsdGeom.XformOp.TypeOrient)

def setTransform(prim, value: Gf.Matrix4d):
    setXformOp(prim, value, UsdGeom.XformOp.TypeTransform)

class VisualArrow(XFormPrim):
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
        name: str = "visual_arrow",
        position: Optional[Sequence[float]] = None,
        translation: Optional[Sequence[float]] = None,
        orientation: Optional[Sequence[float]] = None,
        scale: Optional[Sequence[float]] = None,
        visible: Optional[bool] = True,
        color: Optional[np.ndarray] = None,
        body_radius: Optional[float] = None,
        body_length: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        head_radius: Optional[float] = None,
        head_length: Optional[float] = None,
        visual_material: Optional[VisualMaterial] = None,
    ) -> None:
        if body_radius is None:
            body_radius = 0.1
        if body_length is None:
            body_length = 0.5
        if poll_radius is None:
            poll_radius = 0.02
        if poll_length is None:
            poll_length = 2
        if head_radius is None:
            head_radius = 0.25
        if head_length is None:
            head_length = 0.5
        if visible is None:
            visible = True
        if visual_material is None:
            if color is None:
                color = np.array([0.5, 0.5, 0.5])
            visual_prim_path = find_unique_string_name(
                initial_name="/World/Looks/visual_material", is_unique_fn=lambda x: not is_prim_path_valid(x)
            )
            visual_material = PreviewSurface(prim_path=visual_prim_path, color=color)
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
        body_prim_path = prim_path+"/body"
        if is_prim_path_valid(body_prim_path):
            body_prim = get_prim_at_path(body_prim_path)
            if not body_prim.IsA(UsdGeom.Cylinder):
                raise Exception("The prim at path {} cannot be parsed as an arrow object".format(body_prim_path))
            self.body_geom = UsdGeom.Cylinder(body_prim)
        else:
            self.body_geom = UsdGeom.Cylinder.Define(get_current_stage(), body_prim_path)
            body_prim = get_prim_at_path(body_prim_path)
        poll_prim_path = prim_path+"/poll"
        if is_prim_path_valid(poll_prim_path):
            poll_prim = get_prim_at_path(poll_prim_path)
            if not poll_prim.IsA(UsdGeom.Cylinder):
                raise Exception("The prim at path {} cannot be parsed as an arrow object".format(poll_prim_path))
            self.poll_geom = UsdGeom.Cylinder(poll_prim)
        else:
            self.poll_geom = UsdGeom.Cylinder.Define(get_current_stage(), poll_prim_path)
            poll_prim = get_prim_at_path(poll_prim_path)
        head_prim_path = prim_path+"/head"
        if is_prim_path_valid(head_prim_path):
            head_prim = get_prim_at_path(head_prim_path)
            if not head_prim.IsA(UsdGeom.Cone):
                raise Exception("The prim at path {} cannot be parsed as an arrow object".format(head_prim_path))
            self.head_geom = UsdGeom.Cone(head_prim)
        else:
            self.head_geom = UsdGeom.Cone.Define(get_current_stage(), head_prim_path)
            head_prim = get_prim_at_path(head_prim_path)
        
        if visual_material is not None:
            VisualArrow.apply_visual_material(self, visual_material)
        if body_radius is not None:
            VisualArrow.set_body_radius(self, body_radius)
        if body_length is not None:
            VisualArrow.set_body_length(self, body_length)
        if poll_radius is not None:
            VisualArrow.set_poll_radius(self, poll_radius)
        if poll_length is not None:
            VisualArrow.set_poll_length(self, poll_length)
        if head_radius is not None:
            VisualArrow.set_head_radius(self, head_radius)
        if head_length is not None:
            VisualArrow.set_head_length(self, head_length)
        setTranslate(poll_prim, Gf.Vec3d([0, 0, -poll_length/2]))
        setOrient(poll_prim, Gf.Quatd(0,Gf.Vec3d([0, 0, 0])))
        setScale(poll_prim, Gf.Vec3d([1, 1, 1]))
        setTranslate(body_prim, Gf.Vec3d([body_length/2, 0, 0]))
        setOrient(body_prim, Gf.Quatd(0.707,Gf.Vec3d([0, 0.707, 0])))
        setScale(body_prim, Gf.Vec3d([1, 1, 1]))
        setTranslate(head_prim, Gf.Vec3d([body_length + head_length/2, 0, 0]))
        setOrient(head_prim, Gf.Quatd(0.707,Gf.Vec3d([0, 0.707, 0])))
        setScale(head_prim, Gf.Vec3d([1, 1, 1]))
        radius = VisualArrow.get_body_radius(self)
        height = VisualArrow.get_body_length(self)
        self.body_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -height / 2.0]), Gf.Vec3f([radius, radius, height / 2.0])]
        )
        radius = VisualArrow.get_poll_radius(self)
        height = VisualArrow.get_poll_length(self)
        self.poll_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -height / 2.0]), Gf.Vec3f([radius, radius, height / 2.0])]
        )
        radius = VisualArrow.get_head_radius(self)
        height = VisualArrow.get_head_length(self)
        self.head_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -height / 2.0]), Gf.Vec3f([radius, radius, height / 2.0])]
        )
        return

    def set_body_radius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.body_geom.GetRadiusAttr().Set(radius)
        return

    def get_body_radius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.body_geom.GetRadiusAttr().Get()
    
    def set_body_length(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.body_geom.GetHeightAttr().Set(radius)
        return

    def get_body_length(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.body_geom.GetHeightAttr().Get()
    
    def set_poll_radius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll_geom.GetRadiusAttr().Set(radius)
        return

    def get_poll_radius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll_geom.GetRadiusAttr().Get()
    
    def set_poll_length(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll_geom.GetHeightAttr().Set(radius)
        return

    def get_poll_length(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll_geom.GetHeightAttr().Get()
    
    def set_head_radius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.head_geom.GetRadiusAttr().Set(radius)
        return

    def get_head_radius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.head_geom.GetRadiusAttr().Get()
    
    def set_head_length(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.head_geom.GetHeightAttr().Set(radius)
        return

    def get_head_length(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.head_geom.GetHeightAttr().Get()


class FixedArrow(VisualArrow):
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
        body_radius: Optional[float] = None,
        body_length: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        head_radius: Optional[float] = None,
        head_length: Optional[float] = None,
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
        VisualArrow.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            body_radius = body_radius,
            body_length = body_length,
            poll_radius = poll_radius,
            poll_length = poll_length,
            head_radius = head_radius,
            head_length = head_length,
            visual_material=visual_material,
        )
        #XFormPrim.set_collision_enabled(self, True)
        #if physics_material is not None:
        #    FixedArrow.apply_physics_material(self, physics_material)
        return


class DynamicArrow(RigidPrim, FixedArrow):
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
        body_radius: Optional[float] = None,
        body_length: Optional[float] = None,
        poll_radius: Optional[float] = None,
        poll_length: Optional[float] = None,
        head_radius: Optional[float] = None,
        head_length: Optional[float] = None,
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
        FixedArrow.__init__(
            self,
            prim_path=prim_path,
            name=name,
            position=position,
            translation=translation,
            orientation=orientation,
            scale=scale,
            visible=visible,
            color=color,
            body_radius = body_radius,
            body_length = body_length,
            poll_radius = poll_radius,
            poll_length = poll_length,
            head_radius = head_radius,
            head_length = head_length,
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