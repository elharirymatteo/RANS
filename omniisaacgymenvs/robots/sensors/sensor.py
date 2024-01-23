from omni.isaac.core.utils.rotations import quat_to_rot_matrix
import omni.replicator.core as rep
from omni.isaac.sensor import _sensor
from omni.isaac.sensor import IMUSensor, Camera

import os
import numpy as np
import cv2
import torch

from omniisaacgymenvs.robots.sensors.writer import writer_factory

class RLCamera:
    def __init__(self, prim_path:str, sensor_param:dict):
        self.sensor_param = sensor_param
        #sensor_param["resolution"] does not return list, so expand it and put it in list again.
        self.render_product = rep.create.render_product(
            prim_path, 
            resolution=[*sensor_param["resolution"]])
        self.annotators = {}
        self.writers = {}
        # TODO: make this also parameter.
        self.enable_rgb()
        self.enable_depth()
    
    def enable_rgb(self):
        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach([self.render_product])
        self.annotators.update({"rgb":rgb_annot})
        self.writers.update({"rgb":writer_factory.get("RGBWriter")(add_noise=False)})
    
    def enable_depth(self):
        depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_annot.attach([self.render_product])
        self.annotators.update({"depth":depth_annot})
        self.writers.update({"depth":writer_factory.get("DepthWriter")(add_noise=False)})
    
    def get_observation(self):
        obs_buf = {}
        for modality, annotator in self.annotators.items(): 
            writer = self.writers[modality]
            data_pt = writer.get_data(annotator.get_data())
            obs_buf.update({modality:data_pt})
        return obs_buf

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
sensor_factory.register("RLCamera", RLCamera)
    
class RLSensors:
    """
    Clusters of RLSensors
    """
    def __init__(self, sensor_cfg:dict):
        self.sensors = []
        for sensor_type, sensor_property in sensor_cfg.items():
            sensor = sensor_factory.get(sensor_type)(sensor_property["prim_path"], sensor_property["params"])
            self.sensors.append(sensor)
    
    def get_observation(self):
        obs = {}
        for sensor in self.sensors:
            sensor_obs = sensor.get_observation()
            obs.update(sensor_obs)
        return obs