__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.robots.sensors.exteroceptive.camera_interface import camera_interface_factory
from typing import List
from dataclasses import dataclass, field
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf

import carb

## Replicator hack
carb_settings = carb.settings.get_settings()
carb_settings.set_bool(
    "rtx/raytracing/cached/enabled", 
    False,
)
carb_settings.set_int(
    "rtx/descriptorSets", 
    8192,
)

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
    focalLength: float = None
    focusDistance: float = None
    clippingRange: List[float] = None
    horizontalAperture: float = None
    verticalAperture: float = None
    
@dataclass
class RLCameraParams:
    """
    RLCamera params class.
    Args:
        prim_path (str): path to the prim that the sensor is attached to.
        resolution (List[int]): resolution of the sensor.
        is_override (bool): if True, the sensor parameters will be overriden.
        params (dict): parameters for the sensor.
    """
    prim_path: str
    resolution: List[int]
    is_override: bool
    params: CameraCalibrationParam = field(default_factory=dict)
    
    def __post_init__(self):
        assert len(self.resolution) == 2, f"resolution should be a list of 2 ints, got {self.resolution}"
        self.params = CameraCalibrationParam(**self.params)

class RLCamera:
    """
    RLCamera is a sensor that can be used in RL tasks.
    It uses replicator to record synthetic (mostly images) data.
    """
    def __init__(self, sensor_cfg:dict, rep:object)->None:
        """
        Args:
            sensor_cfg (dict): configuration for the sensor with the following key, value
                prim_path (str): path to the prim that the sensor is attached to
                sensor_param (dict): parameters for the sensor
                override_param (bool): if True, the sensor parameters will be overriden
            rep (object): omni.replicator.core object
        """
        self.sensor_cfg = RLCameraParams(**sensor_cfg)
        self.prim_path = self.sensor_cfg.prim_path
        self.is_override = self.sensor_cfg.is_override
        self.rep = rep

        if self.is_override:
            assert "params" in sensor_cfg.keys(), "params must be provided if override is True."
            self.override_params(get_current_stage(), self.prim_path, self.sensor_cfg.params)
        
        self.render_product = self.rep.create.render_product(
            self.prim_path, 
            resolution=[*self.sensor_cfg.resolution])
        self.annotators = {}
        self.camera_interfaces = {}
        self.enable_rgb()
        self.enable_depth()
    
    def override_params(self, stage, prim_path:str, sensor_param:CameraCalibrationParam)->None:
        """
        Override the sensor parameters if override=True
        Args:
            stage (Stage): stage object
            prim_path (str): path to the prim that the sensor is attached to
            sensor_param (CameraCalibrationParam): parameters for the sensor
        """
        camera = stage.DefinePrim(prim_path, 'Camera')
        camera.GetAttribute('focalLength').Set(sensor_param.focalLength)
        camera.GetAttribute('focusDistance').Set(sensor_param.focusDistance)
        camera.GetAttribute("clippingRange").Set(Gf.Vec2f(*sensor_param.clippingRange))
        camera.GetAttribute("horizontalAperture").Set(sensor_param.horizontalAperture)
        camera.GetAttribute("verticalAperture").Set(sensor_param.verticalAperture)
    
    def enable_rgb(self) -> None:
        """
        Enable RGB as a RL observation
        """
        rgb_annot = self.rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([self.render_product])
        self.annotators.update({"rgb":rgb_annot})
        self.camera_interfaces.update({"rgb":camera_interface_factory.get("RGBInterface")(add_noise=False)})
    
    def enable_depth(self) -> None:
        """
        Enable depth as a RL observation
        """
        depth_annot = self.rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_annot.attach([self.render_product])
        self.annotators.update({"depth":depth_annot})
        self.camera_interfaces.update({"depth":camera_interface_factory.get("DepthInterface")(add_noise=False)})
    
    def get_observation(self) -> dict:
        """
        Returns a dict of observations
        """
        obs_buf = {}
        for modality, annotator in self.annotators.items(): 
            camera_interface = self.camera_interfaces[modality]
            data_pt = camera_interface(annotator.get_data())
            obs_buf.update({modality:data_pt})
        return obs_buf

class CameraFactory:
    """
    Factory class to create sensors.
    """

    def __init__(self):
        self.creators = {}

    def register(self, name: str, sensor):
        """
        Registers a new sensor.
        Args:
            name (str): name of the sensor.
            sensor (object): sensor object.
        """
        self.creators[name] = sensor

    def get(
        self, name: str
    ) -> object:
        """
        Returns a sensor.
        Args:
            name (str): name of the sensor.
        """
        assert name in self.creators.keys(), f"{name} not in {self.creators.keys()}"
        return self.creators[name]

camera_factory = CameraFactory()
camera_factory.register("RLCamera", RLCamera)