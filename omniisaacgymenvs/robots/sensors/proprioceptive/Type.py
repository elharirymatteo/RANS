import numpy as np
import torch
import quaternion
import dataclasses
from typing import List

### sensor typing ###

@dataclasses.dataclass
class Gyroscope_T:
    noise_density: float = 0.0003393695767766752
    random_walk: float = 3.878509448876288e-05
    bias_correlation_time: float = 1.0e3
    turn_on_bias_sigma: float = 0.008726646259971648


@dataclasses.dataclass
class Accelometer_T:
    noise_density: float = 0.004
    random_walk: float = 0.006
    bias_correlation_time: float = 300.0
    turn_on_bias_sigma: float = 0.196

@dataclasses.dataclass
class Sensor_T:
    """
    sensor typing
    dt: physics time resolution
    inertial_to_sensor_frame: transform from inertial frame (ENU) to sensor frame (FLU)
    sensor_frame_to_optical_frame: transform from sensor frame (FLU) to sensor optical optical frame (OPENCV)
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
    gyro_param: Gyroscope_T = Gyroscope_T()
    accel_param: Accelometer_T = Accelometer_T()
    gravity_vector: List[float] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        assert len(self.gravity_vector) == 3
        self.gravity_vector = torch.tensor(self.gravity_vector).to(torch.float32)

@dataclasses.dataclass
class GPS_T(Sensor_T):
    def __post_init__(self):
        super().__post_init__()


### state typing ###
@dataclasses.dataclass
class State:
    """
    state information of any rigid body (to be simulated) respective to inertial frame.
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

    @property
    def body_transform(self):
        """
        transform from inertial frame to body frame.
        """
        transform = torch.zeros(self.position.shape[0], 4, 4).to(self.orientation.device)
        rot_np = self.orientation.cpu().numpy()
        # TODO: find how to use only pytorch to convert quat to mat
        rotation = torch.from_numpy(
            np.stack([quaternion.as_rotation_matrix(np.quaternion(*rot_np[i])) for i in range(rot_np.shape[0])])).to(self.orientation.device).to(torch.float32)
        translation = self.position
        transform[:, :3, :3] = rotation.transpose(1, 2)
        transform[:, :3, 3] = - 1 * torch.bmm(rotation.transpose(1, 2), translation[:, :, None]).squeeze()
        return transform

@dataclasses.dataclass
class ImuState:
    """
    accel and gyro values of body relative to optical frame
    """
    angular_velocity: torch.float32 = torch.zeros(3)
    linear_acceleration: torch.float32 = torch.zeros(3)

    def update(self, angular_velocity, linear_acceleration):
        self.angular_velocity = angular_velocity
        self.linear_acceleration = linear_acceleration