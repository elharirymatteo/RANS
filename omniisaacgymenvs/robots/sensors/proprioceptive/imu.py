__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import numpy as numpy
import torch
from omniisaacgymenvs.robots.sensors.proprioceptive.base_sensor import BaseSensorInterface
from omniisaacgymenvs.robots.sensors.proprioceptive.Type import IMU_T, Accelometer_T, Gyroscope_T, State, ImuState


class IMUInterface(BaseSensorInterface):
    """
    IMU sensor class to simulate accelometer and gyroscope based on pegasus simulator. 
    (https://github.com/PegasusSimulator/PegasusSimulator)
    The way it works is that it takes the state information, directly published from physics engine, 
    and then add imu noise (white noise and time diffusing random walk) to state info. 
    Since it is "inteface", you do not need to call initialize method as seen in omn.isaac.sensor.IMUSensor.
    """
    def __init__(self, sensor_cfg: IMU_T):
        """
        Args:
            sensor_cfg (IMU_T): imu sensor configuration."""
        super().__init__(sensor_cfg)
        self.gravity_vector = self.sensor_cfg["gravity_vector"]
        self._gyroscope_bias = torch.zeros(3, 1)
        self._gyroscope_noise_density = self.sensor_cfg["gyro_param"]["noise_density"]
        self._gyroscope_random_walk = self.sensor_cfg["gyro_param"]["random_walk"]
        self._gyroscope_bias_correlation_time = self.sensor_cfg["gyro_param"][
            "bias_correlation_time"
        ]
        self._gyroscope_turn_on_bias_sigma = self.sensor_cfg["gyro_param"][
            "turn_on_bias_sigma"
        ]
        self._accelerometer_bias = torch.zeros(3, 1)
        self._accelerometer_noise_density = self.sensor_cfg["accel_param"][
            "noise_density"
        ]
        self._accelerometer_random_walk = self.sensor_cfg["accel_param"]["random_walk"]
        self._accelerometer_bias_correlation_time = self.sensor_cfg["accel_param"][
            "bias_correlation_time"
        ]
        self._accelerometer_turn_on_bias_sigma = self.sensor_cfg["accel_param"][
            "turn_on_bias_sigma"
        ]
        self._prev_linear_velocity = None
        self._sensor_state = ImuState()

    
    def update(self, state: State):
        """
        gyroscope and accelerometer simulation (https://ieeexplore.ieee.org/document/7487628)
        gyroscope = angular_velocity + white noise + random walk.
        accelerometer = -1 * (acceleration + white noise + random walk).
        NOTE that accelerometer measures inertial acceleration. Thus, the reading is the negative of body acceleration.
        """
        device = state.angular_velocity.device
        
        # gyroscope term
        tau_g = self._gyroscope_bias_correlation_time
        sigma_g_d = 1 / torch.sqrt(torch.tensor(self.dt)) * self._gyroscope_noise_density
        sigma_b_g = self._gyroscope_random_walk
        sigma_b_g_d = torch.sqrt(-sigma_b_g * sigma_b_g * tau_g / 2.0 * (torch.exp(torch.tensor(-2.0 * self.dt / tau_g)) - 1.0))
        phi_g_d = torch.exp(torch.tensor(-1.0/tau_g * self.dt))
        angular_velocity = torch.bmm(state.body_transform[:, :3, :3], state.angular_velocity[:, :, None]).squeeze()
        for i in range(3):
            self._gyroscope_bias[i] = phi_g_d * self._gyroscope_bias[i] + sigma_b_g_d * torch.randn(1)
            angular_velocity[:, i] = angular_velocity[:, i] + sigma_g_d * torch.randn(1).to(device) + self._gyroscope_bias[i].to(device)
        
        # accelerometer term
        if self._prev_linear_velocity is None:
            self._prev_linear_velocity = torch.zeros_like(state.linear_velocity).to(device)
        tau_a = self._accelerometer_bias_correlation_time
        sigma_a_d = 1.0 / torch.sqrt(torch.tensor(self.dt)) * self._accelerometer_noise_density
        sigma_b_a = self._accelerometer_random_walk
        sigma_b_a_d = torch.sqrt(-sigma_b_a * sigma_b_a * tau_a / 2.0 * (torch.exp(torch.tensor(-2.0 * self.dt / tau_a)) - 1.0))
        phi_a_d = torch.exp(torch.tensor(-1.0 / tau_a * self.dt))
        linear_acceleration_inertial = (state.linear_velocity - self._prev_linear_velocity) / self.dt + self.gravity_vector.to(device)
        self._prev_linear_velocity = state.linear_velocity
        linear_acceleration = torch.bmm(state.body_transform[:, :3, :3], linear_acceleration_inertial[:, :, None]).squeeze()
        for i in range(3):
            self._accelerometer_bias[i] = phi_a_d * self._accelerometer_bias[i] + sigma_b_a_d * torch.randn(1)
            linear_acceleration[:, i] = (
                linear_acceleration[:, i] + sigma_a_d * torch.randn(1).to(device)
                ) #+ self._accelerometer_bias[i]
        
        # transform accel/gyro from body frame to sensor optical frame
        angular_velocity = torch.bmm(self.sensor_frame_to_optical_frame[None, :3, :3].expand(angular_velocity.shape[0], 3, 3).to(device), 
                                     torch.bmm(
                                         self.body_to_sensor_frame[None, :3, :3].expand(angular_velocity.shape[0], 3, 3).to(device), angular_velocity[:, :, None]
                                         )).squeeze()
        linear_acceleration = torch.bmm(self.sensor_frame_to_optical_frame[None, :3, :3].expand(linear_acceleration.shape[0], 3, 3).to(device), 
                                     torch.bmm(
                                         self.body_to_sensor_frame[None, :3, :3].expand(linear_acceleration.shape[0], 3, 3).to(device), linear_acceleration[:, :, None]
                                         )).squeeze()
        self._sensor_state.update(angular_velocity, -1*linear_acceleration)
    
    def reset(self, num_envs: int):
        """
        reset sensor state."""
        self._sensor_state.reset(num_envs=num_envs)

    @property
    def state(self):
        """
        return sensor state."""
        return self._sensor_state

if __name__ == "__main__":
    ## comes from yaml parsed by hydra ##########
    BODY_TO_SENSOR_FRAME = [[1, 0, 0, 0], 
                                      [0, 1, 0, 0], 
                                      [0, 0, 1, 0], 
                                      [0, 0, 0, 1]]
    SENSOR_FRAME_TO_OPTICAL_FRAME = [[-0, -1, 0, 0], 
                                     [0, 0, -1, 0], 
                                     [1, 0, 0, 0], 
                                     [0, 0, 0, 1]]
    GRAVITY_VECTOR = [0, 0, -9.81]
    dt = 0.01
    ACCEL_PARAM = {"noise_density": 0.004, 
                  "random_walk": 0.006, 
                  "bias_correlation_time": 300.0, 
                  "turn_on_bias_sigma": 0.196
                  }
    GYRO_PARAM = {"noise_density": 0.0003393695767766752,
                    "random_walk": 3.878509448876288e-05,
                    "bias_correlation_time": 1.0e3,
                    "turn_on_bias_sigma": 0.008726646259971648
                    }
    #############################################
    imu_t = IMU_T(
        body_to_sensor_frame=BODY_TO_SENSOR_FRAME,
        sensor_frame_to_optical_frame=SENSOR_FRAME_TO_OPTICAL_FRAME, 
        gravity_vector=GRAVITY_VECTOR,
        dt=dt,
        accel_param=Accelometer_T(**ACCEL_PARAM),
        gyro_param=Gyroscope_T(**GYRO_PARAM),
        )
    imu = IMUInterface(imu_t)

    while True:
        N = 16
        position = torch.zeros(N, 3).to(torch.float32)
        orientation = torch.zeros(N, 4).to(torch.float32)
        orientation[:, 0] = 1.0
        linear_velocity = torch.zeros(N, 3).to(torch.float32)
        angular_velocity = torch.zeros(N, 3).to(torch.float32)
        state = State(position, orientation, linear_velocity, angular_velocity)
        imu.update(state)
        print(imu.state)