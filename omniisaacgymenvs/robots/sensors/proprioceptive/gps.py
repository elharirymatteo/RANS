__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import numpy as numpy
from omniisaacgymenvs.robots.sensors.proprioceptive.base_sensor import BaseSensorInterface
from omniisaacgymenvs.robots.sensors.proprioceptive.Type import *

class GPSInterface(BaseSensorInterface):
    """
    GPS sensor class to simulate GPS based on pegasus simulator 
    (https://github.com/PegasusSimulator/PegasusSimulator)
    """
    def __init__(self, sensor_cfg: GPS_T):
        """
        Args:
            sensor_cfg (GPS_T): GPS sensor configuration.
        """
        super().__init__(sensor_cfg)