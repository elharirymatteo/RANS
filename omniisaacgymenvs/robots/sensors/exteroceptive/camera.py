__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import omni.replicator.core as rep

from omniisaacgymenvs.robots.sensors.exteroceptive.camera_interface import camera_interface_factory

class RLCamera:
    """
    RLCamera is a sensor that can be used in RL tasks.
    It uses replicator to record synthetic (mostly images) data.
    """
    def __init__(self, prim_path:str, sensor_param:dict)->None:
        """
        Args:
            prim_path (str): path to the prim that the sensor is attached to
            sensor_param (dict): parameters for the sensor
        """
        self.sensor_param = sensor_param
        self.render_product = rep.create.render_product(
            prim_path, 
            resolution=[*sensor_param["resolution"]])
        self.annotators = {}
        self.camera_interfaces = {}
        self.enable_rgb()
        self.enable_depth()
    
    def enable_rgb(self) -> None:
        """
        Enable RGB as a RL observation"""
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([self.render_product])
        self.annotators.update({"rgb":rgb_annot})
        self.camera_interfaces.update({"rgb":camera_interface_factory.get("RGBInterface")(add_noise=False)})
    
    def enable_depth(self) -> None:
        """
        Enable depth as a RL observation"""
        depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
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
    Factory class to create sensors."""

    def __init__(self):
        self.creators = {}

    def register(self, name: str, sensor):
        """
        Registers a new sensor."""
        self.creators[name] = sensor

    def get(
        self, name: str
    ) -> object:
        """
        Returns a sensor."""
        assert name in self.creators.keys(), f"{name} not in {self.creators.keys()}"
        return self.creators[name]

camera_factory = CameraFactory()
camera_factory.register("RLCamera", RLCamera)