from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from pxr import UsdGeom, UsdPhysics, Sdf, Gf
import os

class D435_Sensor:
    """
    D435 sensor module class. 
    It handles the creation of sensor links(body) and joints between them.
    """
    def __init__(self, cfg:dict):
        """
        Args:
            cfg (dict): configuration for the sensor
        Here are the keys in cfg:
            structure:
                module_name: str
                root_prim:
                    prim_path: str
                    pose: List[float]
                sensor_base:
                    prim_name: str
                    usd_path: str
                links: List[str]
                joints: List[[parent_link_name (str), child_link_name (str), joint_type (str), transform (List[float])]]
            sim:
                RLCamera:
                    prim_path: str
                    rotation: List[float]
                    params:
                        focalLength: float
                        focusDistance: float
                        clippingRange: [float, float]
                        resolution: [float, float]
                        frequency: int
                        horizontalAperture: float
                        verticalAperture: float
        """

        self.cfg = cfg
        self.root_prim_path = cfg["structure"]["root_prim"]["prim_path"]
        self.sensor_base = cfg["structure"]["sensor_base"]
        self.links = cfg["structure"]["links"]
        self.joints = cfg["structure"]["joints"]
        self.sensor_cfg = cfg["sim"]
        self.stage = get_current_stage()

    def _add_root_prim(self) -> None:
        """
        Add root prim."""

        prim = self.stage.DefinePrim(self.root_prim_path, "Xform")
        UsdGeom.Xformable(prim).AddTranslateOp()
        UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(*self.cfg["structure"]["root_prim"]["pose"][:3]))
        prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(*self.cfg["structure"]["root_prim"]["pose"][3:]))

    def attach_to_base(self, source_prim_path:str) -> None:
        """
        Attach the sensor to the base.
        Args:
            source_prim_path (str): path to the prim that the sensor is attached to."""
        
        translation = self.cfg["structure"]["root_prim"]["pose"][:3]
        rotation = self.cfg["structure"]["root_prim"]["pose"][3:]
        rotation_quat = euler_angles_to_quat(euler_angles=rotation, degrees=True)

        fixedJoint = UsdPhysics.FixedJoint.Define(self.stage, os.path.join(self.root_prim_path, self.sensor_base["prim_name"], "camera_attach"))
        fixedJoint.CreateBody0Rel().SetTargets( [Sdf.Path(source_prim_path)])
        fixedJoint.CreateBody1Rel().SetTargets( [Sdf.Path(os.path.join(self.root_prim_path, self.sensor_base["prim_name"]))])
        fixedJoint.CreateLocalPos0Attr().Set(Gf.Vec3f(*translation))
        fixedJoint.CreateLocalRot0Attr().Set(Gf.Quatf(rotation_quat[0], *rotation_quat[1:].tolist()))
        fixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    
    def _add_sensor_link(self) -> None:
        """
        Add sensor link(body)."""

        sensor_link = self.sensor_base["prim_name"]
        sensor_body_usd = os.path.join(os.getcwd(), self.sensor_base["usd_path"])
        prim = self.stage.DefinePrim(os.path.join(self.root_prim_path, sensor_link), "Xform")
        UsdGeom.Xformable(prim).AddTranslateOp()
        UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute('xformOp:translate').Set((0, 0, 0))
        prim.GetAttribute('xformOp:rotateXYZ').Set((0, 0, 0))
        UsdPhysics.RigidBodyAPI.Apply(prim)

        camera_body = add_reference_to_stage(sensor_body_usd, os.path.join(self.root_prim_path, sensor_link, "base_body"))
        UsdGeom.Xformable(camera_body).AddTranslateOp()
        UsdGeom.Xformable(camera_body).AddRotateXYZOp()
        camera_body.GetAttribute('xformOp:translate').Set((0, 0, 0))
        camera_body.GetAttribute('xformOp:rotateXYZ').Set((0, 0, 0))
        UsdPhysics.CollisionAPI.Apply(camera_body)
    
    def _add_link(self, link_name:str) -> None:
        """
        Add link(body).
        Args:
            link_name (str): name of the link."""
        
        self.stage.DefinePrim(os.path.join(self.root_prim_path, link_name), "Xform")

    def _add_transform(self, link_name:str, transform:list) -> None:
        """
        Add transform to the link(body) relative to its parent prim.
        Args:
            link_name (str): name of the link.
            transform (list): transform of the link."""
        
        prim = get_prim_at_path(os.path.join(self.root_prim_path, link_name))
        UsdGeom.Xformable(prim).AddTranslateOp()
        UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute('xformOp:translate').Set(Gf.Vec3f(*transform[:3]))
        prim.GetAttribute('xformOp:rotateXYZ').Set(Gf.Vec3f(*transform[3:]))
        UsdPhysics.RigidBodyAPI.Apply(prim)

    def _add_joint(self, parent_link_name:str, child_link_name:str, joint_name:str, transform:list=[0, 0, 0, 0, 0, 0]) -> None:
        """
        Add joint between two links(bodies).
        Args:
            parent_link_name (str): name of the parent link.
            child_link_name (str): name of the child link.
            joint_name (str): name of the joint.
            transform (list): transform of the joint."""
        
        translate = Gf.Vec3f(*transform[:3])
        rotatation_quat = euler_angles_to_quat(euler_angles=transform[3:], degrees=True)
        rotation = Gf.Quatf(rotatation_quat[0], *rotatation_quat[1:].tolist())

        # define a joint
        fixedJoint = UsdPhysics.FixedJoint.Define(self.stage, os.path.join(self.root_prim_path, self.sensor_base["prim_name"], joint_name))
        fixedJoint.CreateBody0Rel().SetTargets( [Sdf.Path(os.path.join(self.root_prim_path, parent_link_name))])
        fixedJoint.CreateBody1Rel().SetTargets( [Sdf.Path(os.path.join(self.root_prim_path, child_link_name))])

        fixedJoint.CreateLocalPos0Attr().Set(translate)
        fixedJoint.CreateLocalRot0Attr().Set(rotation)
        fixedJoint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixedJoint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
    
    def _add_camera(self) -> None:
        """
        Add usd camera to camera optical link."""

        camera = self.stage.DefinePrim(self.sensor_cfg["RLCamera"]["prim_path"], 'Camera')
        UsdGeom.Xformable(camera).AddTranslateOp()
        UsdGeom.Xformable(camera).AddRotateXYZOp()
        camera.GetAttribute("xformOp:rotateXYZ").Set(Gf.Vec3f(*self.sensor_cfg["RLCamera"]["rotation"]))
        camera.GetAttribute('focalLength').Set(self.sensor_cfg["RLCamera"]["params"]["focalLength"])
        camera.GetAttribute('focusDistance').Set(self.sensor_cfg["RLCamera"]["params"]["focusDistance"])
        camera.GetAttribute("clippingRange").Set(Gf.Vec2f(*self.sensor_cfg["RLCamera"]["params"]["clippingRange"]))
        camera.GetAttribute("horizontalAperture").Set(self.sensor_cfg["RLCamera"]["params"]["horizontalAperture"])
        camera.GetAttribute("verticalAperture").Set(self.sensor_cfg["RLCamera"]["params"]["verticalAperture"])
    
    def initialize(self) -> None:
        """
        Initialize the sensor prim structure."""

        self._add_root_prim()
        self._add_sensor_link()
        for link_name in self.links:
            self._add_link(link_name)
        
        for i, joint in enumerate(self.joints):
            self._add_transform(joint[1], joint[3])
            self._add_joint(joint[0], joint[1], joint[2]+f"_{i}", joint[3])
        
        self._add_camera()

class D455_Sensor(D435_Sensor):
    """
    D455 sensor module class.
    It is identical to D435 exept its extrinsics.
    """
    def __init__(self, cfg:dict):
        """
        Args:
            cfg (dict): configuration for the sensor
        """
        super().__init__(cfg)


class SensorModuleFactory:
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

sensor_module_factory = SensorModuleFactory()
sensor_module_factory.register("D435", D435_Sensor)
sensor_module_factory.register("D455", D455_Sensor)