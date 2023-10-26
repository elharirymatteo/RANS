__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


import math
import torch


class UnevenFloorDisturbance:
    """
    Creates disturbances on the platform by simulating an uneven floor."""

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        # Uneven floor generation
        self._use_uneven_floor = task_cfg["env"]["use_uneven_floor"]
        self._use_sinusoidal_floor = task_cfg["env"]["use_sinusoidal_floor"]
        self._min_freq = task_cfg["env"]["floor_min_freq"]
        self._max_freq = task_cfg["env"]["floor_max_freq"]
        self._min_offset = task_cfg["env"]["floor_min_offset"]
        self._max_offset = task_cfg["env"]["floor_max_offset"]
        self._max_floor_force = task_cfg["env"]["max_floor_force"]
        self._min_floor_force = task_cfg["env"]["min_floor_force"]
        self._max_floor_force = math.sqrt(self._max_floor_force**2 / 2)
        self._min_floor_force = math.sqrt(self._min_floor_force**2 / 2)
        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the uneven floor disturbances."""

        if self._use_sinusoidal_floor:
            self._floor_x_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._floor_y_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._floor_x_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._floor_y_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.floor_forces = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_floor(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Generates the uneven floor."""

        if self._use_sinusoidal_floor:
            self._floor_x_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._floor_y_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._floor_x_offset[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_offset - self._min_offset)
                + self._min_offset
            )
            self._floor_y_offset[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_offset - self._min_offset)
                + self._min_offset
            )
        else:
            r = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * (self._max_floor_force - self._min_floor_force)
                + self._min_floor_force
            )
            theta = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * math.pi
                * 2
            )
            self.floor_forces[env_ids, 0] = torch.cos(theta) * r
            self.floor_forces[env_ids, 1] = torch.sin(theta) * r

    def get_floor_forces(self, root_pos: torch.Tensor) -> None:
        """
        Computes the floor forces for the current state of the robot."""

        if self._use_sinusoidal_floor:
            self.floor_forces[:, 0] = (
                torch.sin(root_pos[:, 0] * self._floor_x_freq + self._floor_x_offset)
                * self._max_floor_force
            )
            self.floor_forces[:, 1] = (
                torch.sin(root_pos[:, 1] * self._floor_y_freq + self._floor_y_offset)
                * self._max_floor_force
            )

        return self.floor_forces


class TorqueDisturbance:
    """
    Creates disturbances on the platform by simulating a torque applied to its center.
    """

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        # Uneven floor generation
        self._use_torque_disturbance = task_cfg["env"]["use_torque_disturbance"]
        self._use_sinusoidal_torque = task_cfg["env"]["use_sinusoidal_torque"]
        self._max_torque = task_cfg["env"]["max_torque"]
        self._min_torque = task_cfg["env"]["min_torque"]

        # use the same min/max frequencies and offsets for the floor
        self._min_freq = task_cfg["env"]["floor_min_freq"]
        self._max_freq = task_cfg["env"]["floor_max_freq"]
        self._min_offset = task_cfg["env"]["floor_min_offset"]
        self._max_offset = task_cfg["env"]["floor_max_offset"]
        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the uneven floor disturbances."""

        if self._use_sinusoidal_torque:
            self._torque_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.torque_forces = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_torque(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Generates the torque disturbance."""

        if self._use_sinusoidal_torque:
            #  use the same min/max frequencies and offsets for the floor
            self._torque_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._torque_offset[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_offset - self._min_offset)
                + self._min_offset
            )
        else:
            r = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * (self._max_torque - self._min_torque)
                + self._min_torque
            )
            # make torques negative for half of the environments at random
            r[
                torch.rand((num_resets), dtype=torch.float32, device=self._device) > 0.5
            ] *= -1
            self.torque_forces[env_ids, 2] = r

    def get_torque_disturbance(self, root_pos: torch.Tensor) -> None:
        """
        Computes the floor forces for the current state of the robot."""

        if self._use_sinusoidal_torque:
            self.torque_forces[:, 2] = (
                torch.sin(root_pos * self._torque_freq + self._torque_offset)
                * self._max_torque
            )

        return self.torque_forces


class NoisyObservations:
    """
    Adds noise to the observations of the robot."""

    def __init__(self, task_cfg: dict) -> None:
        self._add_noise_on_pos = task_cfg["env"]["add_noise_on_pos"]
        self._position_noise_min = task_cfg["env"]["position_noise_min"]
        self._position_noise_max = task_cfg["env"]["position_noise_max"]
        self._add_noise_on_vel = task_cfg["env"]["add_noise_on_vel"]
        self._velocity_noise_min = task_cfg["env"]["velocity_noise_min"]
        self._velocity_noise_max = task_cfg["env"]["velocity_noise_max"]
        self._add_noise_on_heading = task_cfg["env"]["add_noise_on_heading"]
        self._heading_noise_min = task_cfg["env"]["heading_noise_min"]
        self._heading_noise_max = task_cfg["env"]["heading_noise_max"]

    def add_noise_on_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the position of the robot."""

        if self._add_noise_on_pos:
            pos += (
                torch.rand_like(pos)
                * (self._position_noise_max - self._position_noise_min)
                + self._position_noise_min
            )
        return pos

    def add_noise_on_vel(self, vel: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the velocity of the robot."""

        if self._add_noise_on_vel:
            vel += (
                torch.rand_like(vel)
                * (self._velocity_noise_max - self._velocity_noise_min)
                + self._velocity_noise_min
            )
        return vel

    def add_noise_on_heading(self, heading: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the heading of the robot."""

        if self._add_noise_on_heading:
            heading += (
                torch.rand_like(heading)
                * (self._heading_noise_max - self._heading_noise_min)
                + self._heading_noise_min
            )
        return heading


class NoisyActions:
    """
    Adds noise to the actions of the robot."""

    def __init__(self, task_cfg: dict) -> None:
        self._add_noise_on_act = task_cfg["env"]["add_noise_on_act"]
        self._min_action_noise = task_cfg["env"]["min_action_noise"]
        self._max_action_noise = task_cfg["env"]["max_action_noise"]

    def add_noise_on_act(self, act: torch.Tensor):
        """
        Adds noise to the actions of the robot."""

        if self._add_noise_on_act:
            act += (
                torch.rand_like(act) * (self._max_action_noise - self._min_action_noise)
                + self._min_action_noise
            )
        return act
