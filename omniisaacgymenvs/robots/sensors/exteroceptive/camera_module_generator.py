__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf
import os

from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *

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
                links: List[List[str, List[float]]
                camera_sensor:
                    prim_path: str
                    rotation: List[float]
                    params:
                        focalLength: float
                        focusDistance: float
                        clippingRange: [float, float]
                        resolution: [float, float]
                        horizontalAperture: float
                        verticalAperture: float
        """

        self.cfg = cfg
        self.root_prim_path = cfg["structure"]["root_prim"]["prim_path"]
        self.sensor_base = cfg["structure"]["sensor_base"]
        self.links = cfg["structure"]["links"]
        self.stage = get_current_stage()

    def _add_root_prim(self) -> None:
        """
        Add root prim."""

        _, prim = createXform(self.stage, self.root_prim_path)
        setTranslate(prim, Gf.Vec3d(*self.cfg["structure"]["root_prim"]["pose"][:3]))
        setRotateXYZ(prim, Gf.Vec3d(*self.cfg["structure"]["root_prim"]["pose"][3:]))
    
    def _add_sensor_link(self) -> None:
        """
        Add sensor link(body)."""

        _, prim = createXform(self.stage, os.path.join(self.root_prim_path, self.sensor_base["prim_name"]))
        setTranslate(prim, Gf.Vec3d((0, 0, 0)))
        setRotateXYZ(prim, Gf.Vec3d((0, 0, 0)))

        sensor_body_usd = os.path.join(os.getcwd(), self.sensor_base["usd_path"])
        camera_body_prim = add_reference_to_stage(sensor_body_usd, 
                                                  os.path.join(self.root_prim_path, self.sensor_base["prim_name"], "base_body"))
        setTranslate(camera_body_prim, Gf.Vec3d((0, 0, 0)))
        setRotateXYZ(camera_body_prim, Gf.Vec3d((0, 0, 0)))
        applyCollider(camera_body_prim)
    
    def _add_link(self, link_name:str) -> None:
        """
        Add link(body).
        Args:
            link_name (str): name of the link."""
        createXform(self.stage, os.path.join(self.root_prim_path, link_name))

    def _add_transform(self, link_name:str, transform:list) -> None:
        """
        Add transform to the link(body) relative to its parent prim.
        Args:
            link_name (str): name of the link.
            transform (list): transform of the link."""
        
        prim = get_prim_at_path(os.path.join(self.root_prim_path, link_name))
        setTranslate(prim, Gf.Vec3f(*transform[:3]))
        setRotateXYZ(prim, Gf.Vec3f(*transform[3:]))
    
    def _add_camera(self) -> None:
        """
        Add usd camera to camera optical link."""

        camera = self.stage.DefinePrim(self.cfg["structure"]["camera_sensor"]["prim_path"], 'Camera')
        setTranslate(camera, Gf.Vec3d((0, 0, 0)))
        setRotateXYZ(camera, Gf.Vec3f(*self.cfg["structure"]["camera_sensor"]["rotation"]))
        camera.GetAttribute('focalLength').Set(self.cfg["structure"]["camera_sensor"]["params"]["focalLength"])
        camera.GetAttribute('focusDistance').Set(self.cfg["structure"]["camera_sensor"]["params"]["focusDistance"])
        camera.GetAttribute("clippingRange").Set(Gf.Vec2f(*self.cfg["structure"]["camera_sensor"]["params"]["clippingRange"]))
        camera.GetAttribute("horizontalAperture").Set(self.cfg["structure"]["camera_sensor"]["params"]["horizontalAperture"])
        camera.GetAttribute("verticalAperture").Set(self.cfg["structure"]["camera_sensor"]["params"]["verticalAperture"])
    
    def _build_prim_structure(self) -> None:
        """
        Build the sensor prim structure."""

        self._add_root_prim()
        self._add_sensor_link()
        for link in self.links:
            self._add_link(link[0])
            self._add_transform(link[0], link[1])
    
    def build(self) -> None:
        """
        Initialize the sensor prim structure."""

        self._build_prim_structure()
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