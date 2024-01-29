import numpy as numpy
from omniisaacgymenvs.robots.sensors.proprioceptive.base_sensor import BaseSensorInterface
from omniisaacgymenvs.robots.sensors.proprioceptive.Type import *

class GPSInterface(BaseSensorInterface):
    """
    GPS sensor class to simulate GPS based on pegasus simulator 
    (https://github.com/PegasusSimulator/PegasusSimulator)
    """
    def __init__(self, sensor_cfg: GPS_T):
        super().__init__(sensor_cfg)