__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

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
        Args:
            state (State): state of the rigid body to be simulated
        """
        raise NotImplementedError
    
    def reset_idx(self, env_ids:torch.Tensor) -> None:
        """
        reset sensor state of specified env.
        Args:
            env_ids (torch.Tensor): list of env ids to reset
        """
        raise NotImplementedError

    @property
    def state(self):
        """
        return sensor state
        """
        raise NotImplementedError