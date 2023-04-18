# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional

from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
#from omni.isaac.core.materials import PreviewSurface

import numpy as np
import torch
import carb

import omni
import math
from pxr import UsdGeom, Sdf, Gf, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation as SSTR

NUM_THRUSTERS = 4*2# + 6*2 + 8*2

def createXform(stage, path):
    path = omni.usd.get_stage_next_free_path(stage, path, False)
    prim = stage.DefinePrim(path, "Xform")
    return path, prim

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

def refineShape(stage, path, refinement):
    prim = stage.GetPrimAtPath(path)
    prim.CreateAttribute("refinementLevel", Sdf.ValueTypeNames.Int)
    prim.GetAttribute("refinementLevel").Set(refinement)
    prim.CreateAttribute("refinementEnableOverride", Sdf.ValueTypeNames.Bool)
    prim.GetAttribute("refinementEnableOverride").Set(True)

def createSphere(stage, path, radius, refinement):
    sphere_geom = UsdGeom.Sphere.Define(stage, path)
    sphere_geom.GetRadiusAttr().Set(radius)
    refineShape(stage, path, refinement)
    setTranslate(sphere_geom, Gf.Vec3d([0, 0, 0]))
    setOrient(sphere_geom, Gf.Quatd(1, Gf.Vec3d([0, 0, 0])))
    setScale(sphere_geom, Gf.Vec3d([1, 1, 1]))
    return sphere_geom

def addColor(stage, prim, material_path, color): 
    material_path = omni.usd.get_stage_next_free_path(stage, material_path+"/visual_material", False)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path+"/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f(color))
    material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
    arrow_binder = UsdShade.MaterialBindingAPI.Apply(prim)
    arrow_binder.Bind(material)

class CreatePlatform:
    def __init__(self, path):
        self.platform_path = path
        self.thruster_CoM = Gf.Vec3f([0, 0, 0])
        self.thruster_mass = 0.00001
        self.core_CoM = Gf.Vec3f([0, 0, 0])
        self.core_mass = 5.0
        self.refinement = 2
        self.core_radius = 0.5
        self.num_thrusters_per_ring = [4]#, 6, 8]
        self.rings_radius = [0.5]#, 0.75, 1]
        self.thruster_radius = 0.05
        self.thruster_length = 0.1
        self.stage = omni.usd.get_context().get_stage()

    def build(self):    
        platform_path, joints_path = self.createXformArticulationAndJoins()
        core_path = self.createRigidSphere(platform_path + "/core", "sphere", self.core_radius, self.core_CoM, self.core_mass)
        self.createArrowXform(core_path+"/sphere")
        self.createPositionMarkerXform(core_path+"/sphere2")
        dummy_path = self.createRigidSphere(platform_path + "/dummy", "sphere2", 0.00001, self.core_CoM, 0.00001)
        self.createRevoluteJoint(self.stage, joints_path+"/dummy_link", core_path, dummy_path)
        num_thrusters = 0
        for radius, num_ring_thrusters in zip(self.rings_radius, self.num_thrusters_per_ring):
            num_thrusters = self.generateThrusterRing(radius, num_ring_thrusters, num_thrusters, platform_path, joints_path, core_path)

    def createArrowXform(self, path):
        self.arrow_path, self.arrow_prim = createXform(self.stage, path)
        self.createArrowShape(self.stage, self.arrow_path, 0.1, 0.5, [self.core_radius, 0, 0], self.refinement)
        material_path = self.platform_path+"/materials/blue_material"
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, material_path+"/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f([0,0,1.0]))
        material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
        arrow_binder = UsdShade.MaterialBindingAPI.Apply(self.arrow_prim)
        arrow_binder.Bind(material)

    def createPositionMarkerXform(self, path):
        self.marker_path, self.marker_prim = createXform(self.stage, path)
        self.createSphereShape(self.stage, self.marker_path, 0.05, [0, 0, self.core_radius], self.refinement)
        material_path = self.platform_path+"/materials/green_material"
        material = UsdShade.Material.Define(self.stage, material_path)
        shader = UsdShade.Shader.Define(self.stage, material_path+"/shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Float3).Set(Gf.Vec3f([0,1,0]))
        material.CreateSurfaceOutput().ConnectToSource(shader, "surface")
        marker_binder = UsdShade.MaterialBindingAPI.Apply(self.marker_prim)
        marker_binder.Bind(material)

    def createXformArticulationAndJoins(self):
        # Creates the Xform of the platform
        self.platform_path, platform_prim = createXform(self.stage, self.platform_path)
        setTranslate(platform_prim, Gf.Vec3d([0, 0, 0]))
        setOrient(platform_prim, Gf.Quatd(1,Gf.Vec3d([0, 0, 0])))
        setScale(platform_prim, Gf.Vec3d([1, 1, 1]))
        # Creates the Articulation root
        root = UsdPhysics.ArticulationRootAPI.Apply(self.stage.GetPrimAtPath(self.platform_path))
        # Creates an Xform to store the joints in
        joints_path, joints_prim = createXform(self.stage, self.platform_path+'/joints')
        materials_path, materials_prim = createXform(self.stage, self.platform_path+'/materials')
        return self.platform_path, joints_path

    def createRigidSphere(self, path, name, radius, CoM, mass):
        # Creates an Xform to store the sphere
        core_path, core_prim = createXform(self.stage, path)
        setTranslate(core_prim, Gf.Vec3d([0, 0, 0]))
        setOrient(core_prim, Gf.Quatd(1, Gf.Vec3d([0, 0, 0])))
        setScale(core_prim, Gf.Vec3d([1, 1, 1]))
        # Creates a sphere
        sphere_path = core_path+"/"+name
        sphere_geom = createSphere(self.stage, sphere_path, radius, self.refinement)
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        UsdPhysics.RigidBodyAPI.Apply(sphere_prim)
        collider = UsdPhysics.CollisionAPI.Apply(sphere_prim)
        collider.CreateCollisionEnabledAttr(False)
        # Sets the mass and CoM
        massAPI = UsdPhysics.MassAPI.Apply(sphere_prim)
        massAPI.CreateMassAttr().Set(mass)
        return sphere_path

    def generateThrusterRing(self, radius, num_ring_thrusters, num_thrusters, parent_path, joints_path, attachement_path):
        # Create a ring of N thrusters around the platform
        for i in range(num_ring_thrusters):
            # Translate and rotate
            theta = i * 2*math.pi / num_ring_thrusters
            translate = Gf.Vec3d([radius * math.cos(theta), radius * math.sin(theta), 0])
            R1 = SSTR.from_euler('xyz', [math.pi/2, 0, theta])
            Q1 = R1.as_quat()
            R2 = SSTR.from_euler('xyz', [-math.pi/2, 0, theta])
            Q2 = R2.as_quat()
            quat1 = Gf.Quatd(Q1[-1], Gf.Vec3d([Q1[0], Q1[1], Q1[2]]))
            quat2 = Gf.Quatd(Q2[-1], Gf.Vec3d([Q2[0], Q2[1], Q2[2]]))
            # Create thrusters
            self.createThruster(parent_path + "/thruster_" + str(num_thrusters)+"_0", joints_path+"/thruster_joint_"+str(num_thrusters)+"_0", translate, quat1, attachement_path)
            self.createThruster(parent_path + "/thruster_" + str(num_thrusters)+"_1", joints_path+"/thruster_joint_"+str(num_thrusters)+"_1", translate, quat2, attachement_path)
            num_thrusters += 1
        return num_thrusters

    @staticmethod
    def createArrowShape(stage, path, radius, length, offset, refinement):
        length = length / 2
        arrow_body_path = path + "/arrow_body"
        arrow_body_geom = UsdGeom.Cylinder.Define(stage, arrow_body_path)
        arrow_body_geom.GetRadiusAttr().Set(radius)
        arrow_body_geom.GetHeightAttr().Set(length)
        arrow_body_prim = stage.GetPrimAtPath(arrow_body_geom.GetPath())
        setTranslate(arrow_body_geom, Gf.Vec3d([offset[0] + length*0.5, 0, offset[2]]))
        setOrient(arrow_body_geom, Gf.Quatd(0.707, Gf.Vec3d(0, 0.707, 0)))
        setScale(arrow_body_geom, Gf.Vec3d([1, 1, 1]))

        arrow_head_path = path + "/arrow_head"
        arrow_head_geom = UsdGeom.Cone.Define(stage, arrow_head_path)
        arrow_head_geom.GetRadiusAttr().Set(radius*1.5)
        arrow_head_geom.GetHeightAttr().Set(length)
        arrow_head_prim = stage.GetPrimAtPath(arrow_body_geom.GetPath())
        setTranslate(arrow_head_geom, Gf.Vec3d([offset[0] + length*1.5, 0, offset[2]]))
        setOrient(arrow_head_geom, Gf.Quatd(0.707, Gf.Vec3d(0, 0.707, 0)))
        setScale(arrow_head_geom, Gf.Vec3d([1, 1, 1]))

        refineShape(stage, arrow_body_path, refinement)
        refineShape(stage, arrow_head_path, refinement)

    @staticmethod
    def createSphereShape(stage, path, radius, offset, refinement):
        sphere_body_path = path + "/marker"
        sphere_body_geom = UsdGeom.Sphere.Define(stage, sphere_body_path)
        sphere_body_geom.GetRadiusAttr().Set(radius)
        setTranslate(sphere_body_geom, Gf.Vec3d([offset[0], offset[1], offset[2]]))
        setOrient(sphere_body_geom, Gf.Quatd(1.0, Gf.Vec3d(0, 0, 0)))
        setScale(sphere_body_geom, Gf.Vec3d([1, 1, 1]))
        refineShape(stage, sphere_body_path, refinement)

    @staticmethod
    def createThrusterShape(stage, path, radius, height, refinement):
        height /= 2
        # Create the cylinder
        cylinder_path = path + "/cylinder"
        thruster_base_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
        thruster_base_geom.GetRadiusAttr().Set(radius)
        thruster_base_geom.GetHeightAttr().Set(height)
        thruster_base_prim = stage.GetPrimAtPath(thruster_base_geom.GetPath())
        collider = UsdPhysics.CollisionAPI.Apply(thruster_base_prim)
        collider.CreateCollisionEnabledAttr(False)
        # move it such that it stays flush
        setTranslate(thruster_base_geom, Gf.Vec3d([0, 0, height*0.5]))
        setScale(thruster_base_geom, Gf.Vec3d([1, 1, 1]))
        # add a cone to show the direction of the thrust
        cone_path = path + "/cone"
        thruster_cone_geom = UsdGeom.Cone.Define(stage, cone_path)
        thruster_cone_geom.GetRadiusAttr().Set(radius)
        thruster_cone_geom.GetHeightAttr().Set(height)
        thruster_cone_prim = stage.GetPrimAtPath(thruster_cone_geom.GetPath())
        collider = UsdPhysics.CollisionAPI.Apply(thruster_cone_prim)
        collider.CreateCollisionEnabledAttr(False)
        # move and rotate to match reality
        setTranslate(thruster_cone_geom, Gf.Vec3d([0, 0, height*1.5]))
        setRotateXYZ(thruster_cone_geom, Gf.Vec3d([0, 180, 0]))
        setScale(thruster_cone_geom, Gf.Vec3d([1, 1, 1]))
        # Refine
        refineShape(stage, cylinder_path, refinement)
        refineShape(stage, cone_path, refinement)

    @staticmethod
    def createFixedJoint(stage, path, body_path1, body_path2):
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
	
    @staticmethod
    def createRevoluteJoint(stage, path, body_path1, body_path2, axis="Z"):
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
        # Add angular drive for example
        #angularDriveAPI = UsdPhysics.DriveAPI.Apply(stage.GetPrimAtPath(joint.GetPath()), "angular")
        #angularDriveAPI.CreateTypeAttr("force")
        #angularDriveAPI.CreateMaxForceAttr(1e20)
        #angularDriveAPI.CreateDampingAttr(1e10)
        #angularDriveAPI.CreateStiffnessAttr(1e10)

    def createThruster(self, path, joint_path, translate, quat, parent_path):
        # Create Xform
        thruster_path, thruster_prim = createXform(self.stage, path)
        # Move the Xform at the correct position
        setScale(thruster_prim, Gf.Vec3d([1, 1, 1]))
        setTranslate(thruster_prim, translate)
        setOrient(thruster_prim, quat)
        # Add shapes
        self.createThrusterShape(self.stage, thruster_path, self.thruster_radius, self.thruster_length, self.refinement)
        # Make rigid
        UsdPhysics.RigidBodyAPI.Apply(thruster_prim)
        # Add mass
        massAPI = UsdPhysics.MassAPI.Apply(thruster_prim)
        massAPI.CreateCenterOfMassAttr().Set(self.thruster_CoM)
        massAPI.CreateMassAttr().Set(self.thruster_mass)
        # Create joint
        self.createFixedJoint(self.stage, joint_path, parent_path, thruster_path)

class ModularFloatingPlatform(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "modular_floating_platform",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
        scale: Optional[np.array] = None
    ) -> None:
        """[summary]
        """
        
        self._usd_path = usd_path
        self._name = name

        #if self._usd_path is None:
        #    assets_root_path = get_assets_root_path()
        #    if assets_root_path is None:
        #        carb.log_error("Could not find Isaac Sim assets folder")
        #    self._usd_path = "/home/matteo/Projects/OmniIsaacGymEnvs/omniisaacgymenvs/robots/usd/fp3.usd"

        #add_reference_to_stage(self._usd_path, prim_path)
        #scale = torch.tensor([1, 1, 1])
        fp = CreatePlatform(prim_path)
        fp.build()

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale
        )
