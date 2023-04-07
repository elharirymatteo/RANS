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

import numpy as np
import torch
import carb

import omni
import math
from pxr import UsdGeom, Sdf, Gf, UsdPhysics
from scipy.spatial.transform import Rotation as SSTR

NUM_THRUSTERS = 4*2 + 8*2 + 16*2

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
    return sphere_geom

class CreatePlatform:
    def __init__(self, path):
        self.platform_path = path
        self.thruster_CoM = Gf.Vec3f([0, 0, 0])
        self.thruster_mass = 0.0001
        self.core_CoM = Gf.Vec3f([0, 0, 0])
        self.core_mass = 5.0
        self.refinement = 2
        self.core_radius = 0.5
        self.num_thrusters_per_ring = [8, 12, 16]
        self.rings_radius = [0.5, 0.75, 1]
        self.thruster_radius = 0.05
        self.thruster_length = 0.1
        self.stage = omni.usd.get_context().get_stage()

    def build(self):    
        platform_path, joints_path = self.createXformArticulationAndJoins()
        core_path = self.createRigidSphere(platform_path + "/core", self.core_radius, self.core_CoM, self.core_mass/2)
        dummy_path = self.createRigidSphere(platform_path + "/dummy", self.core_radius/2, self.core_CoM, self.core_mass/2)
        self.createRevoluteJoint(self.stage, joints_path+"/dummy_link", core_path, dummy_path)
        num_thrusters = 0
        for radius, num_ring_thrusters in zip(self.rings_radius, self.num_thrusters_per_ring):
            num_thrusters = self.generateThrusterRing(radius, num_ring_thrusters, num_thrusters, platform_path, joints_path, core_path)

    def createXformArticulationAndJoins(self):
        # Creates the Xform of the platform
        platform_path, platform_prim = createXform(self.stage, self.platform_path)
        setTranslate(platform_prim, Gf.Vec3d([0, 0, 0]))
        setOrient(platform_prim, Gf.Quatd(1,Gf.Vec3d([0, 0, 0])))
        setScale(platform_prim, Gf.Vec3d([1, 1, 1]))
        # Creates the Articulation root
        UsdPhysics.ArticulationRootAPI.Apply(self.stage.GetPrimAtPath(self.platform_path))
        # Creates an Xform to store the joints in
        joints_path, joints_prim = createXform(self.stage, platform_path+'/joints')
        return platform_path, joints_path

    def createRigidSphere(self, path, radius, CoM, mass):
        # Creates an Xform to store the sphere
        core_path, core_prim = createXform(self.stage, path)
        setTranslate(core_prim, Gf.Vec3d([0, 0, 0]))
        setOrient(core_prim, Gf.Quatd(1, Gf.Vec3d([0, 0, 0])))
        setScale(core_prim, Gf.Vec3d([1, 1, 1]))
        # Creates a sphere
        sphere_path = core_path+"/sphere"
        createSphere(self.stage, sphere_path, radius, self.refinement)
        UsdPhysics.RigidBodyAPI.Apply(core_prim)
        # Locks to the xy plane.
        core_prim.CreateAttribute("physxRigidBody:lockedPosAxis",  Sdf.ValueTypeNames.Int).Set(4)
        # Sets the mass and CoM
        massAPI = UsdPhysics.MassAPI.Apply(core_prim)
        massAPI.CreateCenterOfMassAttr().Set(CoM)
        massAPI.CreateMassAttr().Set(mass)
        return core_path
    
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
    def createThrusterShape(stage, path, radius, height, refinement):
        height /= 2
        # Create the cylinder
        cylinder_path = path + "/cylinder"
        thruster_base_geom = UsdGeom.Cylinder.Define(stage, cylinder_path)
        thruster_base_geom.GetRadiusAttr().Set(radius)
        thruster_base_geom.GetHeightAttr().Set(height)
        # move it such that it stays flush
        setTranslate(thruster_base_geom, Gf.Vec3d([0, 0, height*0.5]))
        # add a cone to show the direction of the thrust
        cone_path = path + "/cone"
        thruster_cone_geom = UsdGeom.Cone.Define(stage, cone_path)
        thruster_cone_geom.GetRadiusAttr().Set(radius)
        thruster_cone_geom.GetHeightAttr().Set(height)
        # move and rotate to match reality
        setTranslate(thruster_cone_geom, Gf.Vec3d([0, 0, height*1.5]))
        setRotateXYZ(thruster_cone_geom, Gf.Vec3d([0, 180, 0]))
        # Refine
        refineShape(stage, cylinder_path, refinement)
        refineShape(stage, cone_path, refinement)

    @staticmethod
    def createFixedJoint(stage, path, body_path1, body_path2):
        joint = UsdPhysics.FixedJoint.Define(stage, path)
        joint.CreateBody0Rel().SetTargets([body_path1])
        joint.CreateBody1Rel().SetTargets([body_path2])
        joint.CreateBreakForceAttr().Set(1e20)
        joint.CreateBreakTorqueAttr().Set(1e20)
        translate = Gf.Vec3d(stage.GetPrimAtPath(body_path2).GetAttribute('xformOp:translate').Get())
        Q = stage.GetPrimAtPath(body_path2).GetAttribute('xformOp:orient').Get()
        quat0 = Gf.Quatf(Q.GetReal(), Q.GetImaginary()[0], Q.GetImaginary()[1], Q.GetImaginary()[2])
        quat1 = Gf.Quatf(1, 0, 0, 0)
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
        # Set breaking forces
        joint.CreateBreakForceAttr().Set(1e20)
        joint.CreateBreakTorqueAttr().Set(1e20)
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

    def createThruster(self, path, joint_path, translate, quat, parent_path):
        # Create Xform
        thruster_path, thruster_prim = createXform(self.stage, path)
        # Move the Xform at the correct position
        setTranslate(thruster_prim, translate)
        setOrient(thruster_prim, quat)
        # Add shapes
        self.createThrusterShape(self.stage, thruster_path, self.thruster_radius, self.thruster_length, self.refinement)
        # Make rigid
        UsdPhysics.RigidBodyAPI.Apply(thruster_prim)
        thruster_prim.CreateAttribute("physxRigidBody:lockedPosAxis",  Sdf.ValueTypeNames.Int).Set(4)
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
