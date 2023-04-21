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

class Pin:
    def __init__(self, prim_path, ball_radius, poll_radius, poll_length):
        if ball_radius is None:
            ball_radius = 0.1
        if poll_radius is None:
            poll_radius = 0.02
        if poll_length is None:
            poll_length = 2

        self.ball_geom = None
        self.poll_geom = None

        ball_prim_path = prim_path+"/ball"
        if is_prim_path_valid(ball_prim_path):
            ball_prim = get_prim_at_path(ball_prim_path)
            if not ball_prim.IsA(UsdGeom.Sphere):
                raise Exception("The prim at path {} cannot be parsed as a pin object".format(ball_prim_path))
            self.ball_geom = UsdGeom.Sphere(ball_prim)
        else:
            self.ball_geom = UsdGeom.Sphere.Define(get_current_stage(), ball_prim_path)
            ball_prim = get_prim_at_path(ball_prim_path)

        poll_prim_path = prim_path+"/poll"
        if is_prim_path_valid(poll_prim_path):
            poll_prim = get_prim_at_path(poll_prim_path)
            if not poll_prim.IsA(UsdGeom.Cylinder):
                raise Exception("The prim at path {} cannot be parsed as a pin object".format(poll_prim_path))
            self.poll_geom = UsdGeom.Cylinder(poll_prim)
        else:
            self.poll_geom = UsdGeom.Cylinder.Define(get_current_stage(), poll_prim_path)
            poll_prim = get_prim_at_path(poll_prim_path)
  
        setTranslate(poll_prim, Gf.Vec3d([0, 0, -poll_length/2]))
        setOrient(poll_prim, Gf.Quatd(0,Gf.Vec3d([0, 0, 0])))
        setScale(poll_prim, Gf.Vec3d([1, 1, 1]))
        setTranslate(ball_prim, Gf.Vec3d([0, 0, 0]))
        setOrient(ball_prim, Gf.Quatd(1,Gf.Vec3d([0, 0, 0])))
        setScale(ball_prim, Gf.Vec3d([1, 1, 1]))

    def updateExtent(self):
        radius = self.getBallRadius()
        self.ball_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -radius]), Gf.Vec3f([radius, radius, radius])]
        )
        radius = self.getPollRadius()
        height = self.getPollLength()
        self.poll_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -height / 2.0]), Gf.Vec3f([radius, radius, height / 2.0])]
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
    
class Arrow:
    def __init__(self, prim_path, body_radius, body_length, poll_radius, poll_length, head_radius, head_length):
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
        
        setTranslate(poll_prim, Gf.Vec3d([0, 0, -poll_length/2]))
        setOrient(poll_prim, Gf.Quatd(0,Gf.Vec3d([0, 0, 0])))
        setScale(poll_prim, Gf.Vec3d([1, 1, 1]))
        setTranslate(body_prim, Gf.Vec3d([body_length/2, 0, 0]))
        setOrient(body_prim, Gf.Quatd(0.707,Gf.Vec3d([0, 0.707, 0])))
        setScale(body_prim, Gf.Vec3d([1, 1, 1]))
        setTranslate(head_prim, Gf.Vec3d([body_length + head_length/2, 0, 0]))
        setOrient(head_prim, Gf.Quatd(0.707,Gf.Vec3d([0, 0.707, 0])))
        setScale(head_prim, Gf.Vec3d([1, 1, 1]))

    def updateExtent(self):
        radius = self.getBodyRadius()
        height = self.getBodyLength()
        self.body_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -height / 2.0]), Gf.Vec3f([radius, radius, height / 2.0])]
        )
        radius = self.getPollRadius()
        height = self.getPollLength()
        self.poll_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -height / 2.0]), Gf.Vec3f([radius, radius, height / 2.0])]
        )
        radius = self.getHeadRadius()
        height = self.getHeadLength()
        self.head_geom.GetExtentAttr().Set(
            [Gf.Vec3f([-radius, -radius, -height / 2.0]), Gf.Vec3f([radius, radius, height / 2.0])]
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
