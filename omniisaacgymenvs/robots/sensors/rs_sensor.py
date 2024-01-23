from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
import omni.kit.commands
from pxr import UsdGeom, UsdPhysics, Sdf, Gf

import os
import numpy as np

class D435_Sensor:
    def __init__(self, cfg:dict):
        """
        links: list
        joints: list[list(parent, child, joint_type, transform)]
        e.g. 
        links = ["camera_link", "camera_color_frame", "camera_color_optical_frame"]
        joints = [["camera_link", "camera_color_frame", "rigid", [...]], ...]
        """
        self.cfg = cfg
        self.articulation_root_path = cfg["geom"]["articulation_root"]["prim_path"]
        self.sensor_base = cfg["geom"]["sensor_base"]
        self.links = cfg["geom"]["links"]
        self.joints = cfg["geom"]["joints"]
        self.sensor_cfg = cfg["sensor"]
        self.stage = get_current_stage()

    def _add_articulation_root(self):
        prim = self.stage.DefinePrim(self.articulation_root_path, "Xform")
        UsdGeom.Xformable(prim).AddTranslateOp()
        UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(*self.cfg["geom"]["articulation_root"]["pose"][:3]))
        prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(*self.cfg["geom"]["articulation_root"]["pose"][3:]))

    def attach_to_base(self, source_prim_path:str):
        fixedJoint = UsdPhysics.FixedJoint.Define(self.stage, os.path.join(self.articulation_root_path, self.sensor_base["prim_name"], "camera_attach"))
        fixedJoint.CreateBody0Rel().SetTargets( [Sdf.Path(source_prim_path)])
        fixedJoint.CreateBody1Rel().SetTargets( [Sdf.Path(os.path.join(self.articulation_root_path, self.sensor_base["prim_name"]))])

        translation = self.cfg["geom"]["articulation_root"]["pose"][:3]
        rotation = self.cfg["geom"]["articulation_root"]["pose"][3:]
        rotation_quat = euler_angles_to_quat(euler_angles=rotation, degrees=True)
        fixedJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(*translation))
        fixedJoint.CreateLocalRot0Attr().Set(Gf.Quatf(rotation_quat[0], *rotation_quat[1:].tolist()))
        fixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    
    def _add_sensor_link(self):
        # body usd attached to camera_link xform (identity transform)
        sensor_link = self.sensor_base["prim_name"]
        sensor_body_usd = os.path.join(os.getcwd(), self.sensor_base["usd_path"])
        prim = self.stage.DefinePrim(os.path.join(self.articulation_root_path, sensor_link), "Xform")
        UsdGeom.Xformable(prim).AddTranslateOp()
        UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute('xformOp:translate').Set((0, 0, 0))
        prim.GetAttribute('xformOp:rotateXYZ').Set((0, 0, 0))
        UsdPhysics.RigidBodyAPI.Apply(prim)

        camera_body = add_reference_to_stage(sensor_body_usd, os.path.join(self.articulation_root_path, sensor_link, "base_body"))
        UsdGeom.Xformable(camera_body).AddTranslateOp()
        UsdGeom.Xformable(camera_body).AddRotateXYZOp()
        camera_body.GetAttribute('xformOp:translate').Set((0, 0, 0))
        camera_body.GetAttribute('xformOp:rotateXYZ').Set((0, 0, 0))
        UsdPhysics.CollisionAPI.Apply(camera_body)
    
    def _add_link(self, link_name:str):
        self.stage.DefinePrim(os.path.join(self.articulation_root_path, link_name), "Xform")

    def _static_transform(self, link_name:str, transform:list):
        prim = get_prim_at_path(os.path.join(self.articulation_root_path, link_name))
        UsdGeom.Xformable(prim).AddTranslateOp()
        UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(*transform[:3]))
        prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(*transform[3:]))
        UsdPhysics.RigidBodyAPI.Apply(prim)

    def _add_joint(self, parent_link_name:str, child_link_name:str, joint_name:str, transform:list=[0, 0, 0, 0, 0, 0]):
        translate = Gf.Vec3f(*transform[:3])
        rotatation_quat = euler_angles_to_quat(euler_angles=transform[3:], degrees=True)
        rotation = Gf.Quatf(rotatation_quat[0], *rotatation_quat[1:].tolist())

        # define joint
        fixedJoint = UsdPhysics.FixedJoint.Define(self.stage, os.path.join(self.articulation_root_path, self.sensor_base["prim_name"], joint_name))
        fixedJoint.CreateBody0Rel().SetTargets( [Sdf.Path(os.path.join(self.articulation_root_path, parent_link_name))])
        fixedJoint.CreateBody1Rel().SetTargets( [Sdf.Path(os.path.join(self.articulation_root_path, child_link_name))])

        fixedJoint.CreateLocalPos0Attr().Set(translate)
        fixedJoint.CreateLocalRot0Attr().Set(rotation)
        fixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    
    def _add_camera(self):
        camera = self.stage.DefinePrim(self.sensor_cfg["RLCamera"]["prim_path"], 'Camera')
        UsdGeom.Xformable(camera).AddTranslateOp()
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(*self.sensor_cfg["RLCamera"]["rotation"]))
        camera.GetAttribute('focalLength').Set(self.sensor_cfg["RLCamera"]["params"]["focalLength"])
        camera.GetAttribute('focusDistance').Set(self.sensor_cfg["RLCamera"]["params"]["focusDistance"])
        camera.GetAttribute("clippingRange").Set(Gf.Vec2f(*self.sensor_cfg["RLCamera"]["params"]["clippingRange"]))
        camera.GetAttribute("horizontalAperture").Set(self.sensor_cfg["RLCamera"]["params"]["horizontalAperture"])
        camera.GetAttribute("verticalAperture").Set(self.sensor_cfg["RLCamera"]["params"]["verticalAperture"])
    
    def initialize(self):
        """
        [parent_body, child_body, joint_type, transform]
        """
        self._add_articulation_root()
        self._add_sensor_link()
        for link_name in self.links:
            self._add_link(link_name)
        
        for i, joint in enumerate(self.joints):
            self._static_transform(joint[1], joint[3])
            self._add_joint(joint[0], joint[1], joint[2]+f"_{i}", joint[3])
        
        self._add_camera()

    def get_observation(self):
        raise NotImplementedError

# class D435i_Sensor(D435_Sensor):
#     """
#     Since IMU simulation does not work in OIGE, I am commenting this for now.
#     """
#     def __init__(self, cfg:dict):
#         super().__init__(cfg)

#     def _add_imu(self):
#         _, self.imu = omni.kit.commands.execute(
#                 "IsaacSensorCreateImuSensor",
#                 path=self.sensor_cfg["RLIMU"]["prim_path"].split("/")[-1],
#                 parent="/".join(self.sensor_cfg["RLIMU"]["prim_path"].split("/")[:-1]),
#                 sensor_period=-1,
#                 visualize=False,
#             )
    
#     def initialize(self):
#         self._add_articulation_root()
#         self._add_sensor_link()
#         for link_name in self.links:
#             self._add_link(link_name)
        
#         for i, joint in enumerate(self.joints):
#             self._static_transform(joint[1], joint[3])
#             self._add_joint(joint[0], joint[1], joint[2]+f"_{i}", joint[3])
        
#         self._add_camera()
#         self._add_imu()

#     def get_observation(self):
#         raise NotImplementedError

class D455_Sensor(D435_Sensor):
    def __init__(self, cfg:dict):
        super().__init__(cfg)


class SensorFactory:
    """
    Factory class to create tasks."""

    def __init__(self):
        self.creators = {}

    def register(self, name: str, sensor):
        """
        Registers a new task."""
        self.creators[name] = sensor

    def get(
        self, name: str
    ) -> object:
        """
        Returns a task."""
        assert name in self.creators.keys(), f"{name} not in {self.creators.keys()}"
        return self.creators[name]

sensor_factory = SensorFactory()
sensor_factory.register("D435", D435_Sensor)
# sensor_factory.register("D435i", D435i_Sensor)
sensor_factory.register("D455", D455_Sensor)