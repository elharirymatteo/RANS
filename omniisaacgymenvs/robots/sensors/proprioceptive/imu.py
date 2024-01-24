import numpy as numpy
import torch
from omniisaacgymenvs.robots.sensors.proprioceptive.base_sensor import BaseSensor
from omniisaacgymenvs.robots.sensors.proprioceptive.Type import *


class IMU(BaseSensor):
    """
    IMU sensor class to simulate accelometer and gyroscope based on pegasus simulator 
    (https://github.com/PegasusSimulator/PegasusSimulator)
    """
    def __init__(self, sensor_cfg: IMU_T, device="cuda:0"):
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

        self._sensor_state = ImuState()
        self._prev_linear_velocity = None

    
    def update(self, state: State):
        """
        gyroscope and accelerometer simulation (https://ieeexplore.ieee.org/document/7487628)
        NOTE that accelerometer measures inertial acceleration (thus, negative of body acceleration)
        gyroscope = angular_velocity + white noise + random walk
        accelerometer = -1 * (acceleration + white noise + random walk)
        """
        device = state.angular_velocity.device
        
        # gyroscope term
        tau_g = self._gyroscope_bias_correlation_time
        sigma_g_d = 1 / np.sqrt(self.dt) * self._gyroscope_noise_density
        sigma_b_g = self._gyroscope_random_walk
        sigma_b_g_d = np.sqrt(-sigma_b_g * sigma_b_g * tau_g / 2.0 * (np.exp(-2.0 * self.dt / tau_g) - 1.0))
        phi_g_d = np.exp(-1.0/tau_g * self.dt)
        angular_velocity = torch.bmm(state.body_transform[:, :3, :3], state.angular_velocity[:, :, None]).squeeze()
        for i in range(3):
            self._gyroscope_bias[i] = phi_g_d * self._gyroscope_bias[i] + sigma_b_g_d * torch.randn(1)
            angular_velocity[:, i] = angular_velocity[:, i] + sigma_g_d * torch.randn(1).to(device) + self._gyroscope_bias[i].to(device)
        
        # accelerometer term
        if self._prev_linear_velocity is None:
            self._prev_linear_velocity = torch.zeros_like(state.linear_velocity).to(device)
        tau_a = self._accelerometer_bias_correlation_time
        sigma_a_d = 1.0 / np.sqrt(self.dt) * self._accelerometer_noise_density
        sigma_b_a = self._accelerometer_random_walk
        sigma_b_a_d = np.sqrt(-sigma_b_a * sigma_b_a * tau_a / 2.0 * (np.exp(-2.0 * self.dt / tau_a) - 1.0))
        phi_a_d = np.exp(-1.0 / tau_a * self.dt)
        linear_acceleration_inertial = (state.linear_velocity - self._prev_linear_velocity) / self.dt + self.gravity_vector.to(device)
        self._prev_linear_velocity = state.linear_velocity
        linear_acceleration = torch.bmm(state.body_transform[:, :3, :3], linear_acceleration_inertial[:, :, None]).squeeze()
        for i in range(3):
            self._accelerometer_bias[i] = phi_a_d * self._accelerometer_bias[i] + sigma_b_a_d * torch.randn(1)
            linear_acceleration[:, i] = (
                linear_acceleration[:, i] + sigma_a_d * torch.randn(1).to(device)
                ) #+ self._accelerometer_bias[i]
        
        # transform to optical frame
        angular_velocity = torch.bmm(self.sensor_frame_to_optical_frame[None, :3, :3].expand(angular_velocity.shape[0], 3, 3).to(device), 
                                     torch.bmm(
                                         self.body_to_sensor_frame[None, :3, :3].expand(angular_velocity.shape[0], 3, 3).to(device), angular_velocity[:, :, None]
                                         )).squeeze()
        linear_acceleration = torch.bmm(self.sensor_frame_to_optical_frame[None, :3, :3].expand(linear_acceleration.shape[0], 3, 3).to(device), 
                                     torch.bmm(
                                         self.body_to_sensor_frame[None, :3, :3].expand(linear_acceleration.shape[0], 3, 3).to(device), linear_acceleration[:, :, None]
                                         )).squeeze()
        self._sensor_state.update(angular_velocity, -1*linear_acceleration)

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
    imu = IMU(imu_t)

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