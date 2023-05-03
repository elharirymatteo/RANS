import omni

from pxr import Gf, UsdPhysics, UsdGeom, UsdShade, Sdf

# Utils for Xform manipulation
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

def setXformOps(prim, translate:Gf.Vec3d = Gf.Vec3d([0,0,0]),
                      orient: Gf.Quatd = Gf.Quatd(1, Gf.Vec3d([0,0,0])),
                      scale: Gf.Vec3d = Gf.Vec3d([1,1,1])):
    setTranslate(prim, translate)
    setOrient(prim, orient)
    setScale(prim, scale)

def getTransform(prim, parent):
    return UsdGeom.XformCache(0).ComputeRelativeTransform(prim, parent)[0]

# Utils for API manipulation

def applyMaterial(prim, material):
    binder = UsdShade.MaterialBindingAPI.Apply(prim)
    binder.Bind(material)

def applyRigidBody(prim):
    UsdPhysics.RigidBodyAPI.Apply(prim)

def applyCollider(prim, enable=False):
    collider = UsdPhysics.CollisionAPI.Apply(prim)
    collider.CreateCollisionEnabledAttr(enable)

def applyMass(prim, mass, CoM=Gf.Vec3d([0,0,0])):
    massAPI = UsdPhysics.MassAPI.Apply(prim)
    massAPI.CreateMassAttr().Set(mass)
    massAPI.CreateCenterOfMassAttr().Set(CoM)

def createDrive(stage, joint, type="angular", maxforce=1e20, damping=1e20, stifness=1e20):
    # Add angular drive for example
    angularDriveAPI = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(joint.GetPath()), type)
    angularDriveAPI.CreateTypeAttr("force")
    angularDriveAPI.CreateMaxForceAttr(1e20)
    angularDriveAPI.CreateDampingAttr(1e10)
    angularDriveAPI.CreateStiffnessAttr(1e10)

def createXform(stage, path):
    path = omni.usd.get_stage_next_free_path(stage, path, False)
    prim = stage.DefinePrim(path, "Xform")
    setXformOps(prim)
    return path, prim


# Utils for Geom manipulation

def refineShape(stage, path, refinement):
    prim = stage.GetPrimAtPath(path)
    prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
    prim.GetAttribute("refinementLevel").Set(refinement)
    prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool)
    prim.GetAttribute("refinementEnableOverride").Set(True)

def createSphere(stage, path:str, radius:float, refinement:int):
    path = omni.usd.get_stage_next_free_path(stage, path, False)
    sphere_geom = UsdGeom.Sphere.Define(stage, path)
    sphere_geom.GetRadiusAttr().Set(radius)
    setXformOps(sphere_geom)
    refineShape(stage, path, refinement)
    return path, sphere_geom

def createCylinder(stage, path:str, radius:float, height:float, refinement:int):
    path = omni.usd.get_stage_next_free_path(stage, path, False)
    cylinder_geom = UsdGeom.Cylinder.Define(stage, path)
    cylinder_geom.GetRadiusAttr().Set(radius)
    cylinder_geom.GetHeightAttr().Set(height)
    setXformOps(cylinder_geom)
    refineShape(stage, path, refinement)
    return path, cylinder_geom

def createCone(stage, path:str, radius:float, height:float, refinement:int):
    path = omni.usd.get_stage_next_free_path(stage, path, False)
    cone_geom = UsdGeom.Cone.Define(stage, path)
    cone_geom.GetRadiusAttr().Set(radius)
    cone_geom.GetHeightAttr().Set(height)
    setXformOps(cone_geom)
    refineShape(stage, path, refinement)
    return path, cone_geom

def createArrow(stage, path:int, radius:float, length:float, offset:list, refinement:int):
    length = length / 2
    body_path, body_geom = createCylinder(stage, path + "/arrow_body", radius, length, refinement)
    setTranslate(body_geom, Gf.Vec3d([offset[0] + length*0.5, 0, offset[2]]))
    setOrient(body_geom, Gf.Quatd(0.707, Gf.Vec3d(0, 0.707, 0)))
    head_path, head_geom = createCone(stage, path + "/arrow_head", radius*1.5, length, refinement)
    setTranslate(head_geom, Gf.Vec3d([offset[0] + length*1.5, 0, offset[2]]))
    setOrient(head_geom, Gf.Quatd(0.707, Gf.Vec3d(0, 0.707, 0)))

def createThrusterShape(stage, path:str, radius:float, height:float, refinement:int):
    height /= 2
    # Creates a cylinder
    cylinder_path, cylinder_geom = createCylinder(stage, path + "/cylinder", radius, height, refinement)
    cylinder_prim = stage.GetPrimAtPath(cylinder_geom.GetPath())
    applyCollider(cylinder_prim)
    setTranslate(cylinder_geom, Gf.Vec3d([0, 0, height*0.5]))
    setScale(cylinder_geom, Gf.Vec3d([1, 1, 1]))
    # Create a cone
    cone_path, cone_geom = createCone(stage, path + "/cone", radius, height, refinement)
    cone_prim = stage.GetPrimAtPath(cone_geom.GetPath())
    applyCollider(cone_prim)
    setTranslate(cone_geom, Gf.Vec3d([0, 0, height*1.5]))
    setRotateXYZ(cone_geom, Gf.Vec3d([0, 180, 0]))

def createColor(stage, material_path:str, color:list): 
    material_path = omni.usd.get_stage_next_free_path(stage, material_path, False)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path+"/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(color))
    material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
    return material

def createArticulation(stage, path):
    # Creates the Xform of the platform
    path, prim = createXform(stage, path)
    setXformOps(prim)
    # Creates the Articulation root
    root = UsdPhysics.ArticulationRootAPI.Apply(prim)
    return path, prim
    
def createFixedJoint(stage, path:str, body_path1:str, body_path2:str):
    # Create fixed joint
    joint = UsdPhysics.FixedJoint.Define(stage, path)
    # Set body targets
    joint.CreateBody0Rel().SetTargets([body_path1])
    joint.CreateBody1Rel().SetTargets([body_path2])
    # Get from the simulation the position/orientation of the bodies
    translate = Gf.Vec3d(stage.GetPrimAtPath(body_path2).GetAttribute('xformOp:translate').Get())
    Q = stage.GetPrimAtPath(body_path2).GetAttribute('xformOp:orient').Get()
    quat0 = Gf.Quatf(Q.GetReal(), Q.GetImaginary()[0], Q.GetImaginary()[1], Q.GetImaginary()[2])
    quat1 = Gf.Quatf(1, 0, 0, 0)
    # Set the transform between the bodies inside the joint
    joint.CreateLocalPos0Attr().Set(translate)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3d([0, 0, 0]))
    joint.CreateLocalRot0Attr().Set(quat0)
    joint.GetLocalRot1Attr().Set(quat1)
    return joint

def createRevoluteJoint(stage, path:str, body_path1:str, body_path2:str, axis="Z"):
    # Create revolute joint
    joint = UsdPhysics.RevoluteJoint.Define(stage, path)
    # Set body targets
    joint.CreateBody0Rel().SetTargets([body_path1])
    joint.CreateBody1Rel().SetTargets([body_path2])
    # Get from the simulation the position/orientation of the bodies
    translate = Gf.Vec3d(stage.GetPrimAtPath(body_path2).GetAttribute('xformOp:translate').Get())
    Q = stage.GetPrimAtPath(body_path2).GetAttribute('xformOp:orient').Get()
    quat0 = Gf.Quatf(Q.GetReal(), Q.GetImaginary()[0], Q.GetImaginary()[1], Q.GetImaginary()[2])
    quat1 = Gf.Quatf(1, 0, 0, 0)
    # Set the transform between the bodies inside the joint
    joint.CreateLocalPos0Attr().Set(translate)
    joint.CreateLocalPos1Attr().Set(Gf.Vec3d([0, 0, 0]))
    joint.CreateLocalRot0Attr().Set(quat0)
    joint.GetLocalRot1Attr().Set(quat1)
    joint.CreateAxisAttr(axis)
    return joint
