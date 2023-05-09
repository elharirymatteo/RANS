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
import dataclasses

import omni
import math
from pxr import Gf
from scipy.spatial.transform import Rotation as SSTR

from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *


def compute_num_actions(cfg):
    num_actions = 0
    validate_thruster_config(cfg)
    if "thruster_rings" in cfg["thrusters"].keys():
        for thruster_ring in cfg["thrusters"]["thruster_rings"]:
            # Create a ring of N thrusters around the platform
            for i in range(thruster_ring["num_anchors"]):
                num_actions += 1
                if (thruster_ring["style"] == "dual") or (thruster_ring["style"] == "quad"):
                    num_actions += 1
                if thruster_ring["style"] == "quad":
                    num_actions += 1
                    num_actions += 2
    if "thruster_grids" in cfg["thrusters"].keys():
        for thruster_grid in cfg["thrusters"]["thruster_grids"]:
            # Create a grid of N by M thrusters on the platform
            for ix in range(thruster_grid["num_anchors_on_x"]):
                for iy in range(thruster_grid["num_anchors_on_y"]):
                    num_actions += 1
                    if (thruster_grid["style"] == "dual") or (thruster_grid["style"] == "quad"):
                        num_actions += 1
                    if thruster_grid["style"] == "quad":
                        num_actions += 2
    return num_actions


def validate_thruster_config(cfg):
    if "thruster_rings" in cfg["thrusters"].keys():
        for thruster_ring in cfg["thrusters"]["thruster_rings"]:
            # Could use sets here
            keys = ["num_anchors", "ring_offset", "ring_radius",
                    "thruster_radius", "thruster_length", "thruster_mass",
                    "thruster_CoM", "style"]
            for key in keys:
                if key not in thruster_ring.keys():
                    raise ValueError(key+" is missing from one of your thruster configs.")
            extra_keys = ["single", "dual", "quad"]
            if thruster_ring["style"] not in extra_keys:
                raise ValueError("Unknows thruster style, supported styles are: ``single'', ``dual'', ``quad''.")
    
    if "thruster_grids" in cfg["thrusters"].keys():
        for thruster_grid in cfg["thrusters"]["thruster_grids"]:
            keys = ["xmin","xmax","num_anchors_on_x",
                    "ymin","ymax","num_anchors_on_y",
                    "thruster_radius", "thruster_length", "thruster_mass",
                    "thruster_CoM", "style"]
            for key in keys:
                if key not in thruster_grid.keys():
                    raise ValueError(key+" is missing from one of your thruster configs.")
            extra_keys = ["single", "dual", "quad"]
            if thruster_grid["style"] not in extra_keys:
                raise ValueError("Unknows thruster style, supported styles are: ``single'', ``dual'', ``quad''.")

class CreatePlatform:
    def __init__(self, path, cfg):
        self.platform_path = path
        self.joints_path = "joints"
        self.materials_path = "materials"
        self.core_path = None
        self.stage = omni.usd.get_context().get_stage()
        self.num_thrusters = 0
        self.num_actions = 0
        self.thruster_paths = []
        self.transforms2D = []

        self.read_cfg(cfg)

    def read_cfg(self, cfg):
        if "refinement" in cfg.keys():
            self.refinement = cfg["refinement"]
        else:
            self.refinement = 2

        if "core" in cfg.keys():
            if "shape" in cfg["core"].keys():
                self.core_shape = cfg["core"]["shape"]
                assert type(self.core_shape) is str
                self.core_shape.lower()
                assert ((self.core_shape == "sphere") or (self.core_shape == "cylinder"))
            else:
                self.core_shape = "sphere"
            if self.core_shape == "sphere":
                if "radius" in cfg["core"].keys():
                    self.core_radius = cfg["core"]["radius"]
                else:
                    self.core_radius = 0.5
            if self.core_shape == "cylinder":
                if "radius" in cfg["core"].keys():
                    self.core_radius = cfg["core"]["radius"]
                else:
                    self.core_radius = 0.5
                if "height" in cfg["core"].keys():
                    self.height_radius = cfg["core"]["radius"]
                else:
                    self.core_height = 0.5
            if "CoM" in cfg["core"].keys():
                self.core_CoM = Gf.Vec3d(list(cfg["core"]["CoM"]))
            else:
                self.core_CoM = Gf.Vec3d([0,0,0])
            if "Mass" in cfg["core"].keys():
                self.core_mass = cfg["core"]["mass"]
            else:
                self.core_mass = 5.0
        else:
                self.core_shape = "sphere"
                self.core_radius = 0.5
                self.core_CoM = Gf.Vec3d([0,0,0])
                self.core_mass = 5.0

        if "thrusters" in cfg.keys():
            has_thruster = False
            if "thruster_rings" in cfg["thrusters"].keys():
                self.thruster_rings = cfg["thrusters"]["thruster_rings"]
                has_thruster = True
            else:
                self.thruster_rings = []
            if "thruster_grids" in cfg["thrusters"].keys():
                self.thruster_grids = cfg["thrusters"]["thruster_grids"]
                has_thruster = True
            else:
                self.thruster_grids = []
            if "thruster_standalones" in cfg["thrusters"].keys():
                self.thruster_standalones = cfg["thrusters"]["thruster_standalones"]
                has_thruster = True
            else:
                self.thruster_standalones = []
            if not has_thruster:
                raise ValueError("No thruster requested. Generation failed.")
        else:
            self.thruster_standalones = []
            self.thruster_rings = [{"name": "ring1"}]
            self.thruster_grids = []

        if "num_virtual_thrusters" in cfg.keys():
            self.num_virtual_thrusters = cfg["num_virtual_thrusters"]
        else:
            self.num_virtual_thrusters = -1

    def build(self):
        # Creates articulation root and the Xforms to store materials/joints.
        self.platform_path, self.platform_prim = createArticulation(self.stage, self.platform_path)
        self.joints_path, self.joints_prim = createXform(self.stage, self.platform_path + "/" + self.joints_path)
        self.materials_path, self.materials_prim = createXform(self.stage, self.platform_path + "/" + self.materials_path)
        # Creates a set of basic materials
        self.createBasicColors()
        # Creates the main body element and adds the position & heading markers.
        if self.core_shape == "sphere":
            self.core_path = self.createRigidSphere(self.platform_path + "/core", "body", self.core_radius, self.core_CoM, self.core_mass)
            dummy_path = self.createRigidSphere(self.platform_path + "/dummy", "dummy_body", 0.00001, self.core_CoM, 0.00001)
        elif self.core_shape == "cylinder":
            self.core_path = self.createRigidCylinder(self.platform_path + "/core", "body", self.core_radius, self.core_height, self.core_CoM, self.core_mass)
            dummy_path = self.createRigidCylinder(self.platform_path + "/dummy", "dummy_body", 0.00001, 0.00001, self.core_CoM, 0.00001)
        self.createArrowXform(self.core_path+"/arrow")
        self.createPositionMarkerXform(self.core_path+"/marker")
        # Adds a dummy body with a joint & drive so that Isaac stays chill.
        createRevoluteJoint(self.stage, self.joints_path+"/dummy_link", self.core_path, dummy_path)
        # Adds the thrusters
        for thruster_ring in self.thruster_rings:
            self.num_thrusters = self.generateThrusterRing(thruster_ring, self.num_thrusters)
        for thruster_grid in self.thruster_grids:
            self.num_thrusters = self.generateThrusterGrid(thruster_grid, self.num_thrusters)
        for thruster in self.thruster_standalones:
            self.num_thrusters = self.generateThruster(thruster, self.num_thrusters)
        if self.num_virtual_thrusters == -1:
            self.num_virtual_thrusters = self.num_actions
        for i in range(self.num_virtual_thrusters):
            self.createVirtualThruster(self.platform_path + "/v_thruster_"+str(i), self.joints_path + "/v_thruster_joint_"+str(i), self.core_path, 0.0001, Gf.Vec3d([0,0,0]))
        self.transforms2D = np.array(self.transforms2D)

    def createBasicColors(self):
        self.colors = {}
        self.colors["red"] = createColor(self.stage, self.materials_path+"/red", [1,0,0])
        self.colors["green"] = createColor(self.stage, self.materials_path+"/green", [0,1,0])
        self.colors["blue"] = createColor(self.stage, self.materials_path+"/blue", [0,0,1])
        self.colors["white"] = createColor(self.stage, self.materials_path+"/white", [1,1,1])
        self.colors["grey"] = createColor(self.stage, self.materials_path+"/grey", [0.5,0.5,0.5])
        self.colors["dark_grey"] = createColor(self.stage, self.materials_path+"/dark_grey", [0.25,0.25,0.25])
        self.colors["black"] = createColor(self.stage, self.materials_path+"/black", [0,0,0])

    def createArrowXform(self, path: str):
        self.arrow_path, self.arrow_prim = createXform(self.stage, path)
        createArrow(self.stage, self.arrow_path, 0.1, 0.5, [self.core_radius, 0, 0], self.refinement)
        applyMaterial(self.arrow_prim, self.colors["blue"])

    def createPositionMarkerXform(self, path: str):
        self.marker_path, self.marker_prim = createXform(self.stage, path)
        sphere_path, sphere_geom = createSphere(self.stage, self.marker_path+"/marker_sphere", 0.05, self.refinement)
        setTranslate(sphere_geom, Gf.Vec3d([0, 0, self.core_radius]))
        applyMaterial(self.marker_prim, self.colors["green"])

    def createRigidSphere(self, path:str, name:str, radius:float, CoM:list, mass:float):
        # Creates an Xform to store the core body
        path, prim = createXform(self.stage, path)
        # Creates a sphere
        sphere_path = path+"/"+name
        sphere_path, sphere_geom = createSphere(self.stage, path+"/"+name, radius, self.refinement)
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim)
        # Sets the mass and CoM
        applyMass(sphere_prim, mass, CoM)
        return sphere_path
    
    def createRigidCylinder(self, path:str, name:str, radius:float, height:float, CoM:list, mass:float):
        # Creates an Xform to store the core body
        path, prim = createXform(self.stage, path)
        # Creates a sphere
        sphere_path = path+"/"+name
        sphere_path, sphere_geom = createCylinder(self.stage, path+"/"+name, radius, height, self.refinement)
        sphere_prim = self.stage.GetPrimAtPath(sphere_geom.GetPath())
        applyRigidBody(sphere_prim)
        # Sets the collider
        applyCollider(sphere_prim)
        # Sets the mass and CoM
        applyMass(sphere_prim, mass, CoM)
        return sphere_path

    def createVirtualThruster(self, path:str, joint_path:str, parent_path:str, thruster_mass, thruster_CoM):
        # Create Xform
        thruster_path, thruster_prim = createXform(self.stage, path)
        # Add shapes
        setTranslate(thruster_prim, Gf.Vec3d([0,0,0]))
        setOrient(thruster_prim, Gf.Quatd(1, Gf.Vec3d([0,0,0])))
        # Make rigid
        applyRigidBody(thruster_prim)
        # Add mass
        applyMass(thruster_prim, thruster_mass, thruster_CoM)
        # Create joint
        createFixedJoint(self.stage, joint_path, parent_path, thruster_path)
        return thruster_path

    def generateThrusterRing(self, thruster_ring, num_thrusters):
        # Create a ring of N thrusters around the platform
        for i in range(thruster_ring["num_anchors"]):
            # Translate and rotate
            theta = thruster_ring["ring_offset"] + i * 2*math.pi / thruster_ring["num_anchors"]
            translate = Gf.Vec3d([thruster_ring["ring_radius"] * math.cos(theta), thruster_ring["ring_radius"] * math.sin(theta), 0])
            self.transforms2D.append([[np.cos(theta+math.pi/2),np.sin(theta+math.pi/2),0],[np.sin(theta+math.pi/2),-np.cos(theta+math.pi/2),0],[translate[0],translate[1],1]])
            self.num_actions += 1
            if (thruster_ring["style"] == "dual") or (thruster_ring["style"] == "quad"):
                self.transforms2D.append([[np.cos(theta-math.pi/2),np.sin(theta-math.pi/2),0],[np.sin(theta-math.pi/2),-np.cos(theta-math.pi/2),0],[translate[0],translate[1],1]])
                self.num_actions += 1
            if thruster_ring["style"] == "quad":
                self.num_actions += 2
            num_thrusters += 1
        return num_thrusters
    
    def generateThrusterGrid(self, thruster_grid, num_thrusters):
        x_pos = np.linspace(thruster_grid["xmin"], thruster_grid["xmax"], thruster_grid["num_anchors_on_x"])
        y_pos = np.linspace(thruster_grid["ymin"], thruster_grid["ymax"], thruster_grid["num_anchors_on_y"])
        # Create a grid of N by M thrusters on the platform
        for ix in x_pos:
            for iy in y_pos:
                # Translate and rotate
                translate = Gf.Vec3d([ix, iy, 0])
                R1 = SSTR.from_euler('xyz', [math.pi/2, 0, 0])
                self.num_actions += 1
                self.thruster_paths.append(p)
                if (thruster_grid["style"] == "dual") or (thruster_grid["style"] == "quad"):
                    R2 = SSTR.from_euler('xyz', [-math.pi/2, 0, 0])
                    self.num_actions += 1
                self.thruster_paths.append(p)
                if thruster_grid["style"] == "quad":
                    R3 = SSTR.from_euler('xyz', [math.pi/2, 0, math.pi/2])
                    self.thruster_paths.append(p)
                    R4 = SSTR.from_euler('xyz', [-math.pi/2, 0, -math.pi/2])
                    self.thruster_paths.append(p)
                    self.num_actions += 2
                num_thrusters += 1
        return num_thrusters
    
    def generateThruster(self, thruster, num_thrusters):
        translation = Gf.Vec3d([thruster["translation"]["x"],
                                thruster["translation"]["y"],
                                thruster["translation"]["z"]])
        quaternion = Gf.Quatd(thruster["quaternion"]["w"],
                              Gf.Vec3d([thruster["quaternion"]["x"],
                                       thruster["quaternion"]["y"],
                                       thruster["quaternion"]["z"]]))
        self.num_actions += 1
        return num_thrusters
    
    def exportThrusterTransforms(self):
        transforms = []
        core_prim = self.stage.GetPrimAtPath(self.core_path)
        for path in self.thruster_paths:
            prim = self.stage.GetPrimAtPath(path)
            transform = getTransform(prim, core_prim)
            transforms.append(np.array(transform))
        return np.array(transforms)

class ModularFloatingPlatform(Robot):
    def __init__(
        self,
        prim_path: str,
        cfg: dict,
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
        fp = CreatePlatform(prim_path, cfg)
        fp.build()
        self._num_actions = fp.num_actions
        self._transforms = fp.exportThrusterTransforms()
        self._transforms2D = fp.transforms2D

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            scale=scale,
        )
