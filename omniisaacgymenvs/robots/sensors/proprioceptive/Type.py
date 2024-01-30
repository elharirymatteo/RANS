__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import numpy as np
import torch
import dataclasses
from typing import List

### sensor typing ###

@dataclasses.dataclass
class Gyroscope_T:
    """
    Gyroscope typing class.
    """

    noise_density: float = 0.0003393695767766752
    random_walk: float = 3.878509448876288e-05
    bias_correlation_time: float = 1.0e3
    turn_on_bias_sigma: float = 0.008726646259971648


@dataclasses.dataclass
class Accelometer_T:
    """
    Accelometer typing class.
    """

    noise_density: float = 0.004
    random_walk: float = 0.006
    bias_correlation_time: float = 300.0
    turn_on_bias_sigma: float = 0.196

@dataclasses.dataclass
class Sensor_T:
    """
    Sensor typing class.
    Args:
        dt (float): physics time resolution
        inertial_to_sensor_frame (List[float]): transform from inertial frame (ENU) to sensor frame (FLU)
        sensor_frame_to_optical_frame (List[float]): transform from sensor frame (FLU) to sensor optical optical frame (OPENCV)
    """

    dt: float = 0.01
    body_to_sensor_frame: List[float] = dataclasses.field(default_factory=list)
    sensor_frame_to_optical_frame: List[float] = dataclasses.field(default_factory=list)
    def __post_init__(self):
        assert len(self.body_to_sensor_frame) == 4
        assert len(self.sensor_frame_to_optical_frame) == 4
        self.body_to_sensor_frame = torch.tensor(self.body_to_sensor_frame).to(torch.float32)
        self.sensor_frame_to_optical_frame = torch.tensor(self.sensor_frame_to_optical_frame).to(torch.float32)

@dataclasses.dataclass
class IMU_T(Sensor_T):
    """
    IMU typing class.
    Args:
        dt (float): physics time resolution
        inertial_to_sensor_frame (List[float]): transform from inertial frame (ENU) to sensor frame (FLU)
        sensor_frame_to_optical_frame (List[float]): transform from sensor frame (FLU) to sensor optical optical frame (OPENCV)
        gravity_vector (List[float]): gravity vector in inertial frame
        accel_param (Accelometer_T): accelometer parameter
        gyro_param (Gyroscope_T): gyroscope parameter"""

    gyro_param: Gyroscope_T = Gyroscope_T()
    accel_param: Accelometer_T = Accelometer_T()
    gravity_vector: List[float] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.gravity_vector) == 3
        self.gravity_vector = torch.tensor(self.gravity_vector).to(torch.float32)

@dataclasses.dataclass
class GPS_T(Sensor_T):
    """
    GPS typing class.
    Not implemented yet.
    Args:
        dt (float): physics time resolution
        inertial_to_sensor_frame (List[float]): transform from inertial frame (ENU) to sensor frame (FLU)
        sensor_frame_to_optical_frame (List[float]): transform from sensor frame (FLU) to sensor optical optical frame (OPENCV)
    """

    def __post_init__(self):
        super().__post_init__()


### state typing ###
@dataclasses.dataclass
class State:
    """
    State typing class of any rigid body (to be simulated) respective to inertial frame.
    Args:
        position (torch.float32): position of the body in inertial frame.
        orientation (torch.float32): orientation of the body in inertial frame.
        linear_velocity (torch.float32): linear velocity of the body in inertial frame.
        angular_velocity (torch.float32): angular velocity of the body in inertial frame.
    """

    position: torch.float32
    orientation: torch.float32
    linear_velocity: torch.float32
    angular_velocity: torch.float32

    def __post_init__(self):
        assert len(self.position.shape) == 2, f"need to be batched tensor."
        assert len(self.orientation.shape) == 2, f"need to be batched tensor."
        assert len(self.linear_velocity.shape) == 2, f"need to be batched tensor."
        assert len(self.angular_velocity.shape) == 2, f"need to be batched tensor."
    
    @staticmethod
    def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
        """
        Convert batched quaternion to batched rotation matrix.
        Args:
            quat (torch.Tensor): batched quaternion.(..., 4)"""
        EPS = 1e-5
        w, x, y, z = torch.unbind(quat, -1)
        two_s = 2.0 / ((quat * quat).sum(-1) + EPS)
        R = torch.stack(
        (
        1 - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * z + y * w),
        two_s * (x * y + z * w),
        1 - two_s * (x * x + z * z),
        two_s * (y * z - x * w),
        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
        1 - two_s * (x * x + y * y),
        ),
        -1,
        )
        return R.reshape(quat.shape[:-1] + (3, 3))

    @property
    def body_transform(self) -> torch.float32:
        """
        Return transform from inertial frame to body frame(= inverse of body pose).
        T[:, :3, :3] = orientation.T
        T[:, :3, 3] = - orientation.T @ position
        Returns:
            transform (torch.float32): transform matrix from inertial frame to body frame.
        """
        transform = torch.zeros(self.position.shape[0], 4, 4).to(self.orientation.device)
        orientation = self.quat_to_mat(self.orientation)
        transform[:, :3, :3] = orientation.transpose(1, 2)
        transform[:, :3, 3] = - 1 * torch.bmm(orientation.transpose(1, 2), self.position[:, :, None]).squeeze()
        return transform

@dataclasses.dataclass
class ImuState:
    """
    IMU state typing class. 
    Args:
        angular_velocity (torch.float32): angular velocity of the body in body frame.
        linear_acceleration (torch.float32): linear acceleration of the body in body frame.
    """

    angular_velocity: torch.float32 = torch.zeros(1, 3)
    linear_acceleration: torch.float32 = torch.zeros(1, 3)

    def update(self, angular_velocity:torch.float32, linear_acceleration:torch.float32) -> None:
        """
        Update internal attribute from arguments.
        Args:
            angular_velocity (torch.float32): angular velocity of the body in body frame.
            linear_acceleration (torch.float32): linear acceleration of the body in body frame.
        """
        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration
    
    def reset(self, num_envs:int) -> None:
        """
        Reset internal attribute to zero.
        """
        self.angular_velocity = torch.zeros(num_envs, 3)
        self.linear_acceleration = torch.zeros(num_envs, 3)
    
    def reset_idx(self, env_ids:torch.Tensor) -> None:
        """
        Reset internal attribute of specified env to zero.
        """
        self.angular_velocity[env_ids] = 0
        self.linear_acceleration[env_ids] = 0
    
    @property
    def unite_imu(self) -> torch.float32:
        """
        Return IMU state as a single tensor.
        Returns:
            imu (torch.float32): IMU state as a single tensor.
        """
        return torch.cat([self.angular_velocity, self.linear_acceleration], dim=1)