__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import os
from dataclasses import dataclass, field
from omni.isaac.core.utils.stage import get_current_stage, add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from pxr import Gf

from omniisaacgymenvs.robots.articulations.utils.MFP_utils import *

@dataclass
class RootPrimParams:
    """
    Root prim params class.
    Args:
        prim_path (str): path to the prim.
        translation (List[float]): translation of the prim.
        rotation (List[float]): rotation of the prim.
    """
    prim_path: str
    translation: List[float]
    rotation: List[float]
    
    def __post_init__(self):
        assert len(self.translation) == 3, f"translation should be a list of 3 floats, got {self.translation}"
        assert len(self.rotation) == 3, f"rotation should be a list of 3 floats, got {self.rotation}"

@dataclass
class SensorBaseParams:
    """
    Sensor base params class.
    Args:
        prim_name (str): name of the prim.
        usd_path (str): path to the usd file. none if you do not link a usd file.
    """
    prim_name: str = None
    usd_path: str = None

@dataclass
class CameraCalibrationParam:
    """
    Camera calibration params class.
    Args:
        focalLength (float): focal length of the camera.
        focusDistance (float): focus distance of the camera.
        clippingRange (List[float]): clipping range of the camera.
        horizontalAperture (float): horizontal aperture of the camera.
        verticalAperture (float): vertical aperture of the camera.
    """
    focalLength: float
    focusDistance: float
    clippingRange: List[float]
    horizontalAperture: float
    verticalAperture: float

@dataclass
class CameraParams:
    """
    Camera params class.
    Args:
        prim_path (str): path to the prim.
        rotation (List[float]): rotation of the prim.
        params (CameraCalibrationParam): camera calibration params.
    """
    prim_path: str
    rotation: List[float]
    params: CameraCalibrationParam = field(default_factory=dict)
    
    def __post_init__(self):
        assert len(self.rotation) == 3, f"rotation should be a list of 3 floats, got {self.rotation}"
        self.params = CameraCalibrationParam(**self.params)

@dataclass
class CameraModuleParams:
    """
    Camera module params class.
    Args:
        module_name (str): name of the module.
        root_prim (RootPrimParams): root prim params.
        sensor_base (SensorBaseParams): sensor base params.
        links (list): list of links and their transforms.
        camera_sensor (CameraParams): camera params.
    """
    module_name: str
    root_prim: RootPrimParams = field(default_factory=dict)
    sensor_base: SensorBaseParams = field(default_factory=dict)
    links: list = field(default_factory=list)
    camera_sensor: CameraParams = field(default_factory=dict)
    
    def __post_init__(self):
        self.root_prim = RootPrimParams(**self.root_prim)
        self.sensor_base = SensorBaseParams(**self.sensor_base)
        self.camera_sensor = CameraParams(**self.camera_sensor)

class D435_Sensor:
    """
    D435 sensor module class. 
    It handles the creation of sensor links(body) and joints between them.
    """
    def __init__(self, cfg:dict):
        """
        Args:
            cfg (dict): configuration for the sensor
        """

        self.cfg = CameraModuleParams(**cfg)
        self.root_prim_path = self.cfg.root_prim.prim_path
        self.sensor_base = self.cfg.sensor_base
        self.links = self.cfg.links
        self.stage = get_current_stage()

    def _add_root_prim(self) -> None:
        """
        Add root prim.
        """

        _, prim = createXform(self.stage, self.root_prim_path)
        setTranslate(prim, Gf.Vec3d(*self.cfg.root_prim.translation))
        setRotateXYZ(prim, Gf.Vec3d(*self.cfg.root_prim.rotation))
    
    def _add_sensor_link(self) -> None:
        """
        Add sensor link(body).
        If usd file is given, it will be linked to the sensor link.
        """

        _, prim = createXform(self.stage, os.path.join(self.root_prim_path, self.sensor_base.prim_name))
        setTranslate(prim, Gf.Vec3d((0, 0, 0)))
        setRotateXYZ(prim, Gf.Vec3d((0, 0, 0)))
        
        if self.sensor_base.usd_path is not None:
            sensor_body_usd = os.path.join(os.getcwd(), self.sensor_base.usd_path)
            camera_body_prim = add_reference_to_stage(sensor_body_usd, 
                                                    os.path.join(self.root_prim_path, 
                                                                self.sensor_base.prim_name, 
                                                                "base_body"))
            setTranslate(camera_body_prim, Gf.Vec3d((0, 0, 0)))
            setRotateXYZ(camera_body_prim, Gf.Vec3d((0, 0, 0)))
    
    def _add_link(self, link_name:str) -> None:
        """
        Add link(body).
        Args:
            link_name (str): name of the link.
        """
        createXform(self.stage, os.path.join(self.root_prim_path, link_name))

    def _add_transform(self, link_name:str, transform:list) -> None:
        """
        Add transform to the link(body) relative to its parent prim.
        Args:
            link_name (str): name of the link.
            transform (list): transform of the link.
        """
        
        prim = get_prim_at_path(os.path.join(self.root_prim_path, link_name))
        setTranslate(prim, Gf.Vec3f(*transform[:3]))
        setRotateXYZ(prim, Gf.Vec3f(*transform[3:]))
    
    def _add_camera(self) -> None:
        """
        Add usd camera to camera optical link.
        """

        camera = self.stage.DefinePrim(self.cfg.camera_sensor.prim_path, 'Camera')
        setTranslate(camera, Gf.Vec3d((0, 0, 0)))
        setRotateXYZ(camera, Gf.Vec3f(*self.cfg.camera_sensor.rotation))
        camera.GetAttribute('focalLength').Set(self.cfg.camera_sensor.params.focalLength)
        camera.GetAttribute('focusDistance').Set(self.cfg.camera_sensor.params.focusDistance)
        camera.GetAttribute("clippingRange").Set(Gf.Vec2f(*self.cfg.camera_sensor.params.clippingRange))
        camera.GetAttribute("horizontalAperture").Set(self.cfg.camera_sensor.params.horizontalAperture)
        camera.GetAttribute("verticalAperture").Set(self.cfg.camera_sensor.params.verticalAperture)
    
    def _build_prim_structure(self) -> None:
        """
        Build the sensor prim structure.
        """

        self._add_root_prim()
        self._add_sensor_link()
        for link in self.links:
            self._add_link(link[0])
            self._add_transform(link[0], link[1])
    
    def build(self) -> None:
        """
        Initialize the sensor prim structure.
        """

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
    Factory class to create tasks.
    """

    def __init__(self):
        self.creators = {}

    def register(self, name: str, sensor):
        """
        Registers a new task.
        Args:
            name (str): name of the task.
            sensor (object): task object.
        """
        self.creators[name] = sensor

    def get(
        self, name: str
    ) -> object:
        """
        Returns a task.
        Args:
            name (str): name of the task.
        """
        assert name in self.creators.keys(), f"{name} not in {self.creators.keys()}"
        return self.creators[name]

sensor_module_factory = SensorModuleFactory()
sensor_module_factory.register("D435", D435_Sensor)
sensor_module_factory.register("D455", D455_Sensor)