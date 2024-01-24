import numpy as numpy
from base_sensor import BaseSensor
from Type import *

class GPS(BaseSensor):
    """
    GPS sensor class to simulate GPS based on pegasus simulator 
    (https://github.com/PegasusSimulator/PegasusSimulator)
    """
    def __init__(self, sensor_cfg: GPS_T):
        super().__init__(sensor_cfg)