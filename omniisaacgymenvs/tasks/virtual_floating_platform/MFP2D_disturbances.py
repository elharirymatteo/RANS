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
import omni
from typing import Tuple


class MassDistributionDisturbances:
    """
    Creates disturbances on the platform by simulating a mass distribution on the
    platform."""

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored."""
        self._add_mass_disturbances = task_cfg["add_mass_disturbances"]
        self._min_mass = task_cfg["min_mass"]
        self._max_mass = task_cfg["max_mass"]
        self._base_mass = task_cfg["base_mass"]
        print(self._base_mass)
        self._CoM_max_displacement = task_cfg["CoM_max_displacement"]
        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the mass disturbances."""

        self.platforms_mass = (
            torch.ones((self._num_envs, 1), device=self._device, dtype=torch.float32)
            * self._base_mass
        )
        self.platforms_CoM = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def randomize_masses(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Randomizes the masses of the platforms.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""
        if self._add_mass_disturbances:
            self.platforms_mass[env_ids, 0] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_mass - self._min_mass)
                + self._min_mass
            )
            r = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * self._CoM_max_displacement
            )
            theta = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * math.pi
                * 2
            )
            self.platforms_CoM[env_ids, 0] = torch.cos(theta) * r
            self.platforms_CoM[env_ids, 1] = torch.sin(theta) * r

    def get_masses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the masses and CoM of the platforms.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): The masses and CoM of the platforms."""

        return (self.platforms_mass, self.platforms_CoM)

    def set_masses(
        self, body: omni.isaac.core.prims.XFormPrimView, idx: torch.Tensor
    ) -> None:
        """
        Sets the masses and CoM of the platforms.

        Args:
            body (omni.isaac.core.XFormPrimView): The rigid bodies.
            idx (torch.Tensor): The ids of the environments to reset."""
        if self._add_mass_disturbances:
            body.set_masses(self.platforms_mass[idx, 0], indices=idx)
            body.set_coms(self.platforms_CoM[idx], indices=idx)


class UnevenFloorDisturbance:
    """
    Creates disturbances on the platform by simulating an uneven floor."""

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored."""
        self._use_uneven_floor = task_cfg["use_uneven_floor"]
        self._use_sinusoidal_floor = task_cfg["use_sinusoidal_floor"]
        self._min_freq = task_cfg["floor_min_freq"]
        self._max_freq = task_cfg["floor_max_freq"]
        self._min_offset = task_cfg["floor_min_offset"]
        self._max_offset = task_cfg["floor_max_offset"]
        self._max_floor_force = task_cfg["max_floor_force"]
        self._min_floor_force = task_cfg["min_floor_force"]
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
        Generates the uneven floor.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""

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

    def get_floor_forces(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the floor forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The floor forces."""

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
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored."""

        # Uneven floor generation
        self._use_torque_disturbance = task_cfg["use_torque_disturbance"]
        self._use_sinusoidal_torque = task_cfg["use_sinusoidal_torque"]
        self._max_torque = task_cfg["max_torque"]
        self._min_torque = task_cfg["min_torque"]

        # use the same min/max frequencies and offsets for the floor
        self._min_freq = task_cfg["floor_min_freq"]
        self._max_freq = task_cfg["floor_max_freq"]
        self._min_offset = task_cfg["floor_min_offset"]
        self._max_offset = task_cfg["floor_max_offset"]
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
        Generates the torque disturbance.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""

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

    def get_torque_disturbance(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the floor forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The torque disturbance."""

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
        """
        Args:
            task_cfg (dict): The task configuration."""

        self._add_noise_on_pos = task_cfg["add_noise_on_pos"]
        self._position_noise_min = task_cfg["position_noise_min"]
        self._position_noise_max = task_cfg["position_noise_max"]
        self._add_noise_on_vel = task_cfg["add_noise_on_vel"]
        self._velocity_noise_min = task_cfg["velocity_noise_min"]
        self._velocity_noise_max = task_cfg["velocity_noise_max"]
        self._add_noise_on_heading = task_cfg["add_noise_on_heading"]
        self._heading_noise_min = task_cfg["heading_noise_min"]
        self._heading_noise_max = task_cfg["heading_noise_max"]

    def add_noise_on_pos(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the position of the robot.

        Args:
            pos (torch.Tensor): The position of the robot."""

        if self._add_noise_on_pos:
            pos += (
                torch.rand_like(pos)
                * (self._position_noise_max - self._position_noise_min)
                + self._position_noise_min
            )
        return pos

    def add_noise_on_vel(self, vel: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the velocity of the robot.

        Args:
            vel (torch.Tensor): The velocity of the robot.

        Returns:
            torch.Tensor: The velocity of the robot with noise."""

        if self._add_noise_on_vel:
            vel += (
                torch.rand_like(vel)
                * (self._velocity_noise_max - self._velocity_noise_min)
                + self._velocity_noise_min
            )
        return vel

    def add_noise_on_heading(self, heading: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the heading of the robot.

        Args:
            heading (torch.Tensor): The heading of the robot.

        Returns:
            torch.Tensor: The heading of the robot with noise."""

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
        """
        Args:
            task_cfg (dict): The task configuration."""

        self._add_noise_on_act = task_cfg["add_noise_on_act"]
        self._min_action_noise = task_cfg["min_action_noise"]
        self._max_action_noise = task_cfg["max_action_noise"]

    def add_noise_on_act(self, act: torch.Tensor) -> torch.Tensor:
        """
        Adds noise to the actions of the robot.

        Args:
            act (torch.Tensor): The actions of the robot.

        Returns:
            torch.Tensor: The actions of the robot with noise."""

        if self._add_noise_on_act:
            act += (
                torch.rand_like(act) * (self._max_action_noise - self._min_action_noise)
                + self._min_action_noise
            )
        return act
