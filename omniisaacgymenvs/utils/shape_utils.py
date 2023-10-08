__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from pxr import UsdGeom, Gf, UsdShade, Sdf, Usd
import omni
from omni.isaac.core.utils.prims import get_prim_at_path, is_prim_path_valid
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.materials import PreviewSurface
import numpy as np


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


def applyTransforms(prim, translation, rotation, scale, material=None):
    setTranslate(prim, Gf.Vec3d(translation))
    setOrient(prim, Gf.Quatd(rotation[-1], Gf.Vec3d(rotation[:3])))
    setScale(prim, Gf.Vec3d(scale))
    if material is not None:
        applyMaterial(prim, material)


def createPrim(prim_path, name="/body", geom_type=UsdGeom.Cylinder):
    obj_prim_path = prim_path + name
    if is_prim_path_valid(obj_prim_path):
        prim = get_prim_at_path(obj_prim_path)
        if not prim.IsA(geom_type):
            raise Exception(
                "The prim at path {} cannot be parsed as an arrow object".format(
                    obj_prim_path
                )
            )
        geom = geom_type(prim)
    else:
        geom = geom_type.Define(get_current_stage(), obj_prim_path)
        prim = get_prim_at_path(obj_prim_path)
    return geom, prim


def createColor(stage: Usd.Stage, material_path: str, color: list):
    """
    Creates a color material."""

    material_path = omni.usd.get_stage_next_free_path(stage, material_path, False)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(color))
    material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
    return material


def applyMaterial(prim: Usd.Prim, material: UsdShade.Material) -> None:
    """
    Applies a material to a prim."""

    binder = UsdShade.MaterialBindingAPI.Apply(prim)
    binder.Bind(material)


def getCurrentStage():
    return omni.usd.get_context().get_stage()


class Pin:
    def __init__(self, prim_path, ball_radius, poll_radius, poll_length):
        if ball_radius is None:
            ball_radius = 0.1
        if poll_radius is None:
            poll_radius = 0.02
        if poll_length is None:
            poll_length = 2

        self.ball_geom, ball_prim = createPrim(
            prim_path, name="/ball", geom_type=UsdGeom.Sphere
        )
        self.poll_geom, poll_prim = createPrim(
            prim_path, name="/poll", geom_type=UsdGeom.Cylinder
        )

        applyTransforms(poll_prim, [0, 0, -poll_length / 2], [0, 0, 0, 1], [1, 1, 1])
        applyTransforms(ball_prim, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1])

    def updateExtent(self):
        radius = self.getBallRadius()
        self.ball_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        radius = self.getPollRadius()
        height = self.getPollLength()
        self.poll_geom.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )

    def setBallRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.ball_geom.GetRadiusAttr().Set(radius)
        return

    def getBallRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.ball_geom.GetRadiusAttr().Get()

    def setPollRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll_geom.GetRadiusAttr().Set(radius)
        return

    def getPollRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll_geom.GetRadiusAttr().Get()

    def setPollLength(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll_geom.GetHeightAttr().Set(radius)
        return

    def getPollLength(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll_geom.GetHeightAttr().Get()


class Pin3D:
    def __init__(self, prim_path, ball_radius, poll_radius, poll_length):
        if ball_radius is None:
            ball_radius = 0.05
        if poll_radius is None:
            poll_radius = 0.02
        if poll_length is None:
            poll_length = 2

        red_material = createColor(
            getCurrentStage(), "/World/Looks/red_material", [1, 0, 0]
        )
        green_material = createColor(
            getCurrentStage(), "/World/Looks/green_material", [0, 1, 0]
        )
        blue_material = createColor(
            getCurrentStage(), "/World/Looks/blue_material", [0, 0, 1]
        )

        self.ball11_geom, ball11_prim = createPrim(
            prim_path, name="/ball_11", geom_type=UsdGeom.Sphere
        )
        self.ball12_geom, ball12_prim = createPrim(
            prim_path, name="/ball_12", geom_type=UsdGeom.Sphere
        )
        self.ball21_geom, ball21_prim = createPrim(
            prim_path, name="/ball_21", geom_type=UsdGeom.Sphere
        )
        self.ball22_geom, ball22_prim = createPrim(
            prim_path, name="/ball_22", geom_type=UsdGeom.Sphere
        )
        self.ball31_geom, ball31_prim = createPrim(
            prim_path, name="/ball_31", geom_type=UsdGeom.Sphere
        )
        self.ball32_geom, ball32_prim = createPrim(
            prim_path, name="/ball_32", geom_type=UsdGeom.Sphere
        )
        self.poll1_geom, poll1_prim = createPrim(
            prim_path, name="/poll_1", geom_type=UsdGeom.Cylinder
        )
        self.poll2_geom, poll2_prim = createPrim(
            prim_path, name="/poll_2", geom_type=UsdGeom.Cylinder
        )
        self.poll3_geom, poll3_prim = createPrim(
            prim_path, name="/poll_3", geom_type=UsdGeom.Cylinder
        )

        # Z Axis
        applyTransforms(poll1_prim, [0, 0, 0], [0, 0, 0, 1], [1, 1, 1], blue_material)
        applyTransforms(
            ball11_prim, [0, 0, poll_length / 2], [0, 0, 0, 1], [1, 1, 1], blue_material
        )
        applyTransforms(
            ball12_prim,
            [0, 0, -poll_length / 2],
            [0, 0, 0, 1],
            [1, 1, 1],
            blue_material,
        )
        # Y Axis
        applyTransforms(
            poll2_prim, [0, 0, 0], [0.707, 0, 0, 0.707], [1, 1, 1], green_material
        )
        applyTransforms(
            ball21_prim,
            [0, poll_length / 2, 0],
            [0, 0, 0, 1],
            [1, 1, 1],
            green_material,
        )
        applyTransforms(
            ball22_prim,
            [0, -poll_length / 2, 0],
            [0, 0, 0, 1],
            [1, 1, 1],
            green_material,
        )
        # X Axis
        applyTransforms(
            poll3_prim, [0, 0, 0], [0, 0.707, 0, 0.707], [1, 1, 1], red_material
        )
        applyTransforms(
            ball31_prim,
            [poll_length / 2.0, 0, 0],
            [0, 0.707, 0, 0.707],
            [1, 1, 1],
            red_material,
        )
        applyTransforms(
            ball32_prim,
            [-poll_length / 2.0, 0, 0],
            [0, 0.707, 0, 0.707],
            [1, 1, 1],
            red_material,
        )

    def updateExtent(self):
        radius = self.getBallRadius()
        self.ball11_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        self.ball21_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        self.ball31_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        self.ball12_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        self.ball22_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        self.ball32_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        radius = self.getPollRadius()
        height = self.getPollLength()
        self.poll1_geom.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        self.poll2_geom.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        self.poll3_geom.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )

    def setBallRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.ball11_geom.GetRadiusAttr().Set(radius)
        self.ball21_geom.GetRadiusAttr().Set(radius)
        self.ball31_geom.GetRadiusAttr().Set(radius)
        self.ball12_geom.GetRadiusAttr().Set(radius)
        self.ball22_geom.GetRadiusAttr().Set(radius)
        self.ball32_geom.GetRadiusAttr().Set(radius)
        return

    def getBallRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.ball11_geom.GetRadiusAttr().Get()

    def setPollRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll1_geom.GetRadiusAttr().Set(radius)
        self.poll2_geom.GetRadiusAttr().Set(radius)
        self.poll3_geom.GetRadiusAttr().Set(radius)
        return

    def getPollRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll1_geom.GetRadiusAttr().Get()

    def setPollLength(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll1_geom.GetHeightAttr().Set(radius)
        self.poll2_geom.GetHeightAttr().Set(radius)
        self.poll3_geom.GetHeightAttr().Set(radius)
        return

    def getPollLength(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll1_geom.GetHeightAttr().Get()


class Arrow:
    def __init__(
        self,
        prim_path,
        body_radius,
        body_length,
        poll_radius,
        poll_length,
        head_radius,
        head_length,
    ):
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

        # createPrim()
        self.body_geom, body_prim = createPrim(
            prim_path, name="/body", geom_type=UsdGeom.Cylinder
        )
        self.poll_geom, poll_prim = createPrim(
            prim_path, name="/poll", geom_type=UsdGeom.Cylinder
        )
        self.head_geom, head_prim = createPrim(
            prim_path, name="/head", geom_type=UsdGeom.Cone
        )

        applyTransforms(poll_prim, [0, 0, -poll_length / 2], [0, 0, 0, 1], [1, 1, 1])
        applyTransforms(
            body_prim, [body_length / 2, 0, 0], [0, 0.707, 0, 0.707], [1, 1, 1]
        )
        applyTransforms(
            head_prim,
            [body_length + head_length / 2, 0, 0],
            [0, 0.707, 0, 0.707],
            [1, 1, 1],
        )

    def updateExtent(self):
        radius = self.getBodyRadius()
        height = self.getBodyLength()
        self.body_geom.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        radius = self.getPollRadius()
        height = self.getPollLength()
        self.poll_geom.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        radius = self.getHeadRadius()
        height = self.getHeadLength()
        self.head_geom.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        return

    def setBodyRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.body_geom.GetRadiusAttr().Set(radius)
        return

    def getBodyRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.body_geom.GetRadiusAttr().Get()

    def setBodyLength(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.body_geom.GetHeightAttr().Set(radius)
        return

    def getBodyLength(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.body_geom.GetHeightAttr().Get()

    def setPollRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll_geom.GetRadiusAttr().Set(radius)
        return

    def getPollRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll_geom.GetRadiusAttr().Get()

    def setPollLength(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.poll_geom.GetHeightAttr().Set(radius)
        return

    def getPollLength(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.poll_geom.GetHeightAttr().Get()

    def setHeadRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.head_geom.GetRadiusAttr().Set(radius)
        return

    def getHeadRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.head_geom.GetRadiusAttr().Get()

    def setHeadLength(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.head_geom.GetHeightAttr().Set(radius)
        return

    def getHeadLength(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.head_geom.GetHeightAttr().Get()


class Arrow3D:
    def __init__(self, prim_path, body_radius, body_length, head_radius, head_length):
        if body_radius is None:
            body_radius = 0.1
        if body_length is None:
            body_length = 0.5
        if head_radius is None:
            head_radius = 0.25
        if head_length is None:
            head_length = 0.5

        red_material = createColor(
            getCurrentStage(), "/World/Looks/red_material", [1, 0, 0]
        )
        green_material = createColor(
            getCurrentStage(), "/World/Looks/green_material", [0, 1, 0]
        )
        blue_material = createColor(
            getCurrentStage(), "/World/Looks/blue_material", [0, 0, 1]
        )

        # createPrim()
        self.body_geom1, body_prim1 = createPrim(
            prim_path, name="/body1", geom_type=UsdGeom.Cylinder
        )
        self.body_geom2, body_prim2 = createPrim(
            prim_path, name="/body2", geom_type=UsdGeom.Cylinder
        )
        self.body_geom3, body_prim3 = createPrim(
            prim_path, name="/body3", geom_type=UsdGeom.Cylinder
        )
        self.head_geom1, head_prim1 = createPrim(
            prim_path, name="/head1", geom_type=UsdGeom.Cone
        )
        self.head_geom2, head_prim2 = createPrim(
            prim_path, name="/head2", geom_type=UsdGeom.Cone
        )
        self.head_geom3, head_prim3 = createPrim(
            prim_path, name="/head3", geom_type=UsdGeom.Cone
        )

        # Z Axis
        applyTransforms(
            body_prim1,
            [0, 0, body_length / 2],
            [0, 0, 0, 1.0],
            [1, 1, 1],
            material=blue_material,
        )
        applyTransforms(
            head_prim1,
            [0, 0, body_length + head_length / 2],
            [0, 0, 0, 1.0],
            [1, 1, 1],
            material=blue_material,
        )
        # Y Axis
        applyTransforms(
            body_prim2,
            [0, body_length / 2, 0],
            [0.707, 0, 0, 0.707],
            [1, 1, 1],
            material=green_material,
        )
        applyTransforms(
            head_prim2,
            [0, body_length + head_length / 2, 0],
            [-0.707, 0, 0, 0.707],
            [1, 1, 1],
            material=green_material,
        )
        # X Axis
        applyTransforms(
            body_prim3,
            [body_length / 2, 0, 0],
            [0, 0.707, 0, 0.707],
            [1, 1, 1],
            material=red_material,
        )
        applyTransforms(
            head_prim3,
            [body_length + head_length / 2, 0, 0],
            [0, 0.707, 0, 0.707],
            [1, 1, 1],
            material=red_material,
        )

    def updateExtent(self):
        radius = self.getBodyRadius()
        height = self.getBodyLength()
        self.body_geom1.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        self.body_geom2.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        self.body_geom3.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        radius = self.getHeadRadius()
        height = self.getHeadLength()
        self.head_geom1.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        self.head_geom2.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        self.head_geom3.GetExtentAttr().Set(
            [
                Gf.Vec3f([-radius, -radius, -height / 2.0]),
                Gf.Vec3f([radius, radius, height / 2.0]),
            ]
        )
        return

    def setBodyRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.body_geom1.GetRadiusAttr().Set(radius)
        self.body_geom2.GetRadiusAttr().Set(radius)
        self.body_geom3.GetRadiusAttr().Set(radius)
        return

    def getBodyRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.body_geom1.GetRadiusAttr().Get()

    def setBodyLength(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.body_geom1.GetHeightAttr().Set(radius)
        self.body_geom2.GetHeightAttr().Set(radius)
        self.body_geom3.GetHeightAttr().Set(radius)
        return

    def getBodyLength(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.body_geom1.GetHeightAttr().Get()

    def setHeadRadius(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.head_geom1.GetRadiusAttr().Set(radius)
        self.head_geom2.GetRadiusAttr().Set(radius)
        self.head_geom3.GetRadiusAttr().Set(radius)
        return

    def getHeadRadius(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.head_geom1.GetRadiusAttr().Get()

    def setHeadLength(self, radius: float) -> None:
        """[summary]

        Args:
            radius (float): [description]
        """
        self.head_geom1.GetHeightAttr().Set(radius)
        self.head_geom2.GetHeightAttr().Set(radius)
        self.head_geom3.GetHeightAttr().Set(radius)
        return

    def getHeadLength(self) -> float:
        """[summary]

        Returns:
            float: [description]
        """
        return self.head_geom1.GetHeightAttr().Get()
