import numpy as np
from dataclasses import asdict
from omniisaacgymenvs.robots.sensors.proprioceptive.Type import *


class BaseSensorInterface:
    """
    Base sensor class
    """
    def __init__(self, sensor_cfg: Sensor_T):
        """
        dt: float
        inertial_to_sensor_frame: List[float]
        sensor_frame_to_optical_frame: List[float]
        """
        self.sensor_cfg = asdict(sensor_cfg)
        self.dt = self.sensor_cfg["dt"]
        self.body_to_sensor_frame = self.sensor_cfg["body_to_sensor_frame"]
        self.sensor_frame_to_optical_frame = self.sensor_cfg[
            "sensor_frame_to_optical_frame"
        ]
        self._sensor_state = None

    def update(self, state: State):
        """
        state is the state of the rigid body to be simulated
        """
        raise NotImplementedError

    @property
    def state(self):
        return self._sensor_state