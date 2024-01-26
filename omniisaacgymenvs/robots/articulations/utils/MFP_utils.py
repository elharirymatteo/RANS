import omni

from typing import List, Tuple
from pxr import Gf, UsdPhysics, UsdGeom, UsdShade, Sdf, Usd

# ==================================================================================================
# Utils for Xform manipulation
# ==================================================================================================


def setXformOp(prim: Usd.Prim, value, property: UsdGeom.XformOp.Type) -> None:
    """
    Sets a transform operatios on a prim.

    Args:
        prim (Usd.Prim): The prim to set the transform operation.
        value: The value of the transform operation.
        property (UsdGeom.XformOp.Type): The type of the transform operation.
    """

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


def setScale(prim: Usd.Prim, value: Gf.Vec3d) -> None:
    """
    Sets the scale of a prim.

    Args:
        prim (Usd.Prim): The prim to set the scale.
        value (Gf.Vec3d): The value of the scale.
    """

    setXformOp(prim, value, UsdGeom.XformOp.TypeScale)


def setTranslate(prim: Usd.Prim, value: Gf.Vec3d) -> None:
    """
    Sets the translation of a prim.

    Args:
        prim (Usd.Prim): The prim to set the translation.
        value (Gf.Vec3d): The value of the translation.
    """

    setXformOp(prim, value, UsdGeom.XformOp.TypeTranslate)


def setRotateXYZ(prim: Usd.Prim, value: Gf.Vec3d) -> None:
    """
    Sets the rotation of a prim.

    Args:
        prim (Usd.Prim): The prim to set the rotation.
        value (Gf.Vec3d): The value of the rotation.
    """

    setXformOp(prim, value, UsdGeom.XformOp.TypeRotateXYZ)


def setOrient(prim: Usd.Prim, value: Gf.Quatd) -> None:
    """
    Sets the rotation of a prim.

    Args:
        prim (Usd.Prim): The prim to set the rotation.
        value (Gf.Quatd): The value of the rotation.
    """

    setXformOp(prim, value, UsdGeom.XformOp.TypeOrient)


def setTransform(prim, value: Gf.Matrix4d) -> None:
    """
    Sets the transform of a prim.

    Args:
        prim (Usd.Prim): The prim to set the transform.
        value (Gf.Matrix4d): The value of the transform.
    """

    setXformOp(prim, value, UsdGeom.XformOp.TypeTransform)


def setXformOps(
    prim,
    translate: Gf.Vec3d = Gf.Vec3d([0, 0, 0]),
    orient: Gf.Quatd = Gf.Quatd(1, Gf.Vec3d([0, 0, 0])),
    scale: Gf.Vec3d = Gf.Vec3d([1, 1, 1]),
) -> None:
    """
    Sets the transform of a prim.

    Args:
        prim (Usd.Prim): The prim to set the transform.
        translate (Gf.Vec3d): The value of the translation.
        orient (Gf.Quatd): The value of the rotation.
        scale (Gf.Vec3d): The value of the scale.
    """

    setTranslate(prim, translate)
    setOrient(prim, orient)
    setScale(prim, scale)


def getTransform(prim: Usd.Prim, parent: Usd.Prim) -> Gf.Matrix4d:
    """
    Gets the transform of a prim relative to its parent.

    Args:
        prim (Usd.Prim): The prim to get the transform.
        parent (Usd.Prim): The parent of the prim.
    """

    return UsdGeom.XformCache(0).ComputeRelativeTransform(prim, parent)[0]


# ==================================================================================================
# Utils for API manipulation
# ==================================================================================================


def applyMaterial(
    prim: Usd.Prim, material: UsdShade.Material
) -> UsdShade.MaterialBindingAPI:
    """
    Applies a material to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the material.
        material (UsdShade.Material): The material to apply.

    Returns:
        UsdShade.MaterialBindingAPI: The MaterialBindingAPI.
    """

    binder = UsdShade.MaterialBindingAPI.Apply(prim)
    binder.Bind(material)
    return binder


def applyRigidBody(prim: Usd.Prim) -> UsdPhysics.RigidBodyAPI:
    """
    Applies a RigidBodyAPI to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the RigidBodyAPI.

    Returns:
        UsdPhysics.RigidBodyAPI: The RigidBodyAPI.
    """

    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    return rigid


def applyCollider(prim: Usd.Prim, enable: bool = False) -> UsdPhysics.CollisionAPI:
    """
    Applies a ColliderAPI to a prim.

    Args:
        prim (Usd.Prim): The prim to apply the ColliderAPI.
        enable (bool): Enable or disable the collider.

    Returns:
        UsdPhysics.CollisionAPI: The ColliderAPI.
    """

    collider = UsdPhysics.CollisionAPI.Apply(prim)
    collider.CreateCollisionEnabledAttr(enable)
    return collider


def applyMass(
    prim: Usd.Prim, mass: float, CoM: Gf.Vec3d = Gf.Vec3d([0, 0, 0])
) -> UsdPhysics.MassAPI:
    """
    Applies a MassAPI to a prim.
    Sets the mass and the center of mass of the prim.

    Args:
        prim (Usd.Prim): The prim to apply the MassAPI.
        mass (float): The mass of the prim.
        CoM (Gf.Vec3d): The center of mass of the prim.

    Returns:
        UsdPhysics.MassAPI: The MassAPI.
    """

    massAPI = UsdPhysics.MassAPI.Apply(prim)
    massAPI.CreateMassAttr().Set(mass)
    massAPI.CreateCenterOfMassAttr().Set(CoM)
    return massAPI


def createDrive(
    joint: Usd.Prim,
    token: str = "transX",
    damping: float = 1e3,
    stiffness: float = 1e6,
) -> UsdPhysics.DriveAPI:
    """
    Creates a DriveAPI on a joint.

    Args:
        joint (Usd.Prim): The joint to apply the DriveAPI.
        token (str): The type of the drive.
        damping (float): The damping of the drive.
        stiffness (float): The stiffness of the drive.

    Returns:
        UsdPhysics.DriveAPI: The DriveAPI.
    """

    driveAPI = UsdPhysics.DriveAPI.Apply(joint, token)
    driveAPI.CreateTypeAttr("force")
    driveAPI.CreateDampingAttr(damping)
    driveAPI.CreateStiffnessAttr(stiffness)
    return driveAPI


def createLimit(
    joint: Usd.Prim,
    token: str = "transX",
    low: float = None,
    high: float = None,
) -> UsdPhysics.LimitAPI:
    """
    Creates a LimitAPI on a joint.

    Args:
        joint (Usd.Prim): The joint to apply the LimitAPI.
        token (str): The type of the limit.
        low (float): The lower limit of the joint.
        high (float): The upper limit of the joint.

    Returns:
        UsdPhysics.LimitAPI: The LimitAPI.
    """

    limitAPI = UsdPhysics.LimitAPI.Apply(joint, token)
    if low:
        limitAPI.CreateLowAttr(low)
    if high:
        limitAPI.CreateHighAttr(high)
    return limitAPI


# ==================================================================================================
# Utils for Geom manipulation
# ==================================================================================================


def createXform(
    stage: Usd.Stage,
    path: str,
) -> Tuple[str, Usd.Prim]:
    """
    Creates an Xform prim.
    And sets the default transform operations.

    Args:
        stage (Usd.Stage): The stage to create the Xform prim.
        path (str): The path of the Xform prim.

    Returns:
        Tuple[str, Usd.Prim]: The path and the prim of the Xform prim.
    """

    path = omni.usd.get_stage_next_free_path(stage, path, False)
    prim = stage.DefinePrim(path, "Xform")
    setXformOps(prim)
    return path, prim


def refineShape(stage: Usd.Stage, path: str, refinement: int) -> None:
    """
    Refines the geometry of a shape.
    This operation is purely visual, it does not affect the physics simulation.

    Args:
        stage (Usd.Stage): The stage to refine the shape.
        path (str): The path of the shape.
        refinement (int): The number of times to refine the shape.
    """

    prim = stage.GetPrimAtPath(path)
    prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
    prim.GetAttribute("refinementLevel").Set(refinement)
    prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool)
    prim.GetAttribute("refinementEnableOverride").Set(True)


def createSphere(
    stage: Usd.Stage,
    path: str,
    radius: float,
    refinement: int,
) -> Tuple[str, UsdGeom.Sphere]:
    """
    Creates a sphere.

    Args:
        stage (Usd.Stage): The stage to create the sphere.
        path (str): The path of the sphere.
        radius (float): The radius of the sphere.
        refinement (int): The number of times to refine the sphere.

    Returns:
        Tuple[str, UsdGeom.Sphere]: The path and the prim of the sphere.
    """

    path = omni.usd.get_stage_next_free_path(stage, path, False)
    sphere_geom = UsdGeom.Sphere.Define(stage, path)
    sphere_geom.GetRadiusAttr().Set(radius)
    setXformOps(sphere_geom)
    refineShape(stage, path, refinement)
    return path, sphere_geom


def createCylinder(
    stage: Usd.Stage,
    path: str,
    radius: float,
    height: float,
    refinement: int,
) -> Tuple[str, UsdGeom.Cylinder]:
    """
    Creates a cylinder.

    Args:
        stage (Usd.Stage): The stage to create the cylinder.
        path (str): The path of the cylinder.
        radius (float): The radius of the cylinder.
        height (float): The height of the cylinder.
        refinement (int): The number of times to refine the cylinder.

    Returns:
        Tuple[str, UsdGeom.Cylinder]: The path and the prim of the cylinder.
    """

    path = omni.usd.get_stage_next_free_path(stage, path, False)
    cylinder_geom = UsdGeom.Cylinder.Define(stage, path)
    cylinder_geom.GetRadiusAttr().Set(radius)
    cylinder_geom.GetHeightAttr().Set(height)
    setXformOps(cylinder_geom)
    refineShape(stage, path, refinement)
    return path, cylinder_geom


def createCone(
    stage: Usd.Stage,
    path: str,
    radius: float,
    height: float,
    refinement: int,
) -> Tuple[str, UsdGeom.Cone]:
    """
    Creates a cone.

    Args:
        stage (Usd.Stage): The stage to create the cone.
        path (str): The path of the cone.
        radius (float): The radius of the cone.
        height (float): The height of the cone.
        refinement (int): The number of times to refine the cone.

    Returns:
        Tuple[str, UsdGeom.Cone]: The path and the prim of the cone.
    """

    path = omni.usd.get_stage_next_free_path(stage, path, False)
    cone_geom = UsdGeom.Cone.Define(stage, path)
    cone_geom.GetRadiusAttr().Set(radius)
    cone_geom.GetHeightAttr().Set(height)
    setXformOps(cone_geom)
    refineShape(stage, path, refinement)
    return path, cone_geom


def createArrow(
    stage: Usd.Stage,
    path: int,
    radius: float,
    length: float,
    offset: list,
    refinement: int,
) -> None:
    """
    Creates an arrow.

    Args:
        stage (Usd.Stage): The stage to create the arrow.
        path (str): The path of the arrow.
        radius (float): The radius of the arrow.
        length (float): The length of the arrow.
        offset (list): The offset of the arrow.
        refinement (int): The number of times to refine the arrow.

    Returns:
        Tuple[str, UsdGeom.Cone]: The path and the prim of the arrow.
    """

    length = length / 2
    body_path, body_geom = createCylinder(
        stage, path + "/arrow_body", radius, length, refinement
    )
    setTranslate(body_geom, Gf.Vec3d([offset[0] + length * 0.5, 0, offset[2]]))
    setOrient(body_geom, Gf.Quatd(0.707, Gf.Vec3d(0, 0.707, 0)))
    head_path, head_geom = createCone(
        stage, path + "/arrow_head", radius * 1.5, length, refinement
    )
    setTranslate(head_geom, Gf.Vec3d([offset[0] + length * 1.5, 0, offset[2]]))
    setOrient(head_geom, Gf.Quatd(0.707, Gf.Vec3d(0, 0.707, 0)))


def createThrusterShape(
    stage: Usd.Stage,
    path: str,
    radius: float,
    height: float,
    refinement: int,
) -> None:
    """
    Creates a thruster.

    Args:
        stage (Usd.Stage): The stage to create the thruster.
        path (str): The path of the thruster.
        radius (float): The radius of the thruster.
        height (float): The height of the thruster.
        refinement (int): The number of times to refine the thruster.

    Returns:
        Tuple[str, UsdGeom.Cone]: The path and the prim of the thruster.
    """

    height /= 2
    # Creates a cylinder
    cylinder_path, cylinder_geom = createCylinder(
        stage, path + "/cylinder", radius, height, refinement
    )
    cylinder_prim = stage.GetPrimAtPath(cylinder_geom.GetPath())
    applyCollider(cylinder_prim)
    setTranslate(cylinder_geom, Gf.Vec3d([0, 0, height * 0.5]))
    setScale(cylinder_geom, Gf.Vec3d([1, 1, 1]))
    # Create a cone
    cone_path, cone_geom = createCone(stage, path + "/cone", radius, height, refinement)
    cone_prim = stage.GetPrimAtPath(cone_geom.GetPath())
    applyCollider(cone_prim)
    setTranslate(cone_geom, Gf.Vec3d([0, 0, height * 1.5]))
    setRotateXYZ(cone_geom, Gf.Vec3d([0, 180, 0]))


def createColor(
    stage: Usd.Stage,
    material_path: str,
    color: list,
) -> UsdShade.Material:
    """
    Creates a color material.

    Args:
        stage (Usd.Stage): The stage to create the color material.
        material_path (str): The path of the material.
        color (list): The color of the material

    Returns:
        UsdShade.Material: The material.
    """

    material_path = omni.usd.get_stage_next_free_path(stage, material_path, False)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(color))
    material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
    return material


def createArticulation(
    stage: Usd.Stage,
    path: str,
) -> Tuple[str, Usd.Prim]:
    """
    Creates an ArticulationRootAPI on a prim.

    Args:
        stage (Usd.Stage): The stage to create the ArticulationRootAPI.
        path (str): The path of the ArticulationRootAPI.

    Returns:
        Tuple[str, Usd.Prim]: The path and the prim of the ArticulationRootAPI.
    """

    # Creates the Xform of the platform
    path, prim = createXform(stage, path)
    setXformOps(prim)
    # Creates the Articulation root
    root = UsdPhysics.ArticulationRootAPI.Apply(prim)
    return path, prim


def createFixedJoint(
    stage: Usd.Stage,
    path: str,
    body_path1: str,
    body_path2: str,
) -> UsdPhysics.FixedJoint:
    """
    Creates a fixed joint between two bodies.

    Args:
        stage (Usd.Stage): The stage to create the fixed joint.
        path (str): The path of the fixed joint.
        body_path1 (str): The path of the first body.
        body_path2 (str): The path of the second body.

    Returns:
        UsdPhysics.FixedJoint: The fixed joint.
    """

    # Create fixed joint
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    # Set body targets
    joint.CreateBody0Rel().SetTargets([body_path1])
    joint.CreateBody1Rel().SetTargets([body_path2])
    # Get from the simulation the position/orientation of the bodies
    translate = Gf.Vec3d(
        stage.GetPrimAtPath(body_path2).GetAttribute("xformOp:translate").Get()
    )
    Q = stage.GetPrimAtPath(body_path2).GetAttribute("xformOp:orient").Get()
    quat0 = Gf.Quatf(
        Q.GetReal(), Q.GetImaginary()[0], Q.GetImaginary()[1], Q.GetImaginary()[2]
    )
    # Set the transform between the bodies inside the joint
    joint.CreateLocalPos0Attr().Set(translate)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3d([0, 0, 0]))
    joint.CreateLocalRot0Attr().Set(quat0)
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    return joint


def createRevoluteJoint(
    stage: Usd.Stage,
    path: str,
    body_path1: str,
    body_path2: str,
    axis="Z",
    enable_drive: bool = False,
) -> UsdPhysics.RevoluteJoint:
    """
    Creates a revolute joint between two bodies.

    Args:
        stage (Usd.Stage): The stage to create the revolute joint.
        path (str): The path of the revolute joint.
        body_path1 (str): The path of the first body.
        body_path2 (str): The path of the second body.
        axis (str): The axis of rotation.
        enable_drive (bool): Enable or disable the drive.

    Returns:
        UsdPhysics.RevoluteJoint: The revolute joint.
    """

    # Create revolute joint
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)

    # Set body targets
    joint.CreateBody0Rel().SetTargets([body_path1])
    joint.CreateBody1Rel().SetTargets([body_path2])

    # Get from the simulation the position/orientation of the bodies
    body_1_prim = stage.GetPrimAtPath(body_path1)
    body_2_prim = stage.GetPrimAtPath(body_path2)
    xform_body_1 = UsdGeom.Xformable(body_1_prim)
    xform_body_2 = UsdGeom.Xformable(body_2_prim)
    transform_body_1 = xform_body_1.ComputeLocalToWorldTransform(0.0)
    transform_body_2 = xform_body_2.ComputeLocalToWorldTransform(0.0)
    t12 = np.matmul(np.linalg.inv(transform_body_1), transform_body_2)
    translate_body_12 = Gf.Vec3f([t12[3][0], t12[3][1], t12[3][2]])
    Q_body_12 = Gf.Transform(Gf.Matrix4d(t12.tolist())).GetRotation().GetQuat()

    # Set the transform between the bodies inside the joint
    joint.CreateLocalPos0Attr().Set(translate_body_12)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3d([0, 0, 0]))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(Q_body_12))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1, 0, 0, 0))
    joint.CreateAxisAttr(axis)
    return joint
