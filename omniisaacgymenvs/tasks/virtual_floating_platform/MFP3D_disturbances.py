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

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_disturbances import (
    UnevenFloorDisturbance as UnevenFloorDisturbance2D,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_disturbances import (
    TorqueDisturbance as TorqueDisturbance2D,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_disturbances import (
    NoisyActions as NoisyActions2D,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_disturbances import (
    NoisyObservations as NoisyObservations2D,
)
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_disturbances import (
    MassDistributionDisturbances as MassDistributionDisturbances2D,
)


class MassDistributionDisturbances(MassDistributionDisturbances2D):
    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        super(MassDistributionDisturbances, self).__init__(task_cfg, num_envs, device)

    def randomize_masses(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Randomizes the masses of the platforms.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""

        self.platforms_mass[env_ids] = (
            torch.rand(num_resets, dtype=torch.float32, device=self._device)
            * (self._max_mass - self._min_mass)
            + self._min_mass
        ).unsqueeze(1)
        r = (
            torch.rand((num_resets), dtype=torch.float32, device=self._device)
            * self._CoM_max_displacement
        )
        theta = (
            torch.rand((num_resets), dtype=torch.float32, device=self._device)
            * math.pi
            * 2
        )
        phi = (
            torch.rand((num_resets), dtype=torch.float32, device=self._device) * math.pi
        )
        self.platforms_CoM[env_ids, 0] = torch.cos(theta) * r
        self.platforms_CoM[env_ids, 1] = torch.sin(theta) * r
        self.platforms_CoM[env_ids, 2] = torch.cos(phi) * r


class UnevenFloorDisturbance(UnevenFloorDisturbance2D):
    """
    Creates disturbances on the platform by simulating an uneven floor."""

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments to create.
            device (str): The device to use for the computation."""

        super(UnevenFloorDisturbance, self).__init__(task_cfg, num_envs, device)
        self._max_floor_force = math.sqrt((self._max_floor_force**2) / 3)
        self._min_floor_force = math.sqrt((self._min_floor_force**2) / 3)

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
            self._floor_z_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._floor_x_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._floor_y_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._floor_z_offset = torch.zeros(
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
            self._floor_z_freq[env_ids] = (
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
            self._floor_z_offset[env_ids] = (
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
            phi = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * math.pi
            )
            self.floor_forces[env_ids, 0] = torch.cos(theta) * r
            self.floor_forces[env_ids, 1] = torch.sin(theta) * r
            self.floor_forces[env_ids, 2] = torch.cos(phi) * r

    def get_floor_forces(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the floor forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot."""

        if self._use_sinusoidal_floor:
            self.floor_forces[:, 0] = (
                torch.sin(root_pos[:, 0] * self._floor_x_freq + self._floor_x_offset)
                * self._max_floor_force
            )
            self.floor_forces[:, 1] = (
                torch.sin(root_pos[:, 1] * self._floor_y_freq + self._floor_y_offset)
                * self._max_floor_force
            )
            self.floor_forces[:, 2] = (
                torch.sin(root_pos[:, 2] * self._floor_z_freq + self._floor_z_offset)
                * self._max_floor_force
            )

        return self.floor_forces


class TorqueDisturbance(TorqueDisturbance2D):
    """
    Creates disturbances on the platform by simulating a torque applied to its center.
    """

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        super(TorqueDisturbance, self).__init__(task_cfg, num_envs, device)
        self._max_torque = math.sqrt((self._max_torque**2) / 3)
        self._min_torque = math.sqrt((self._min_torque**2) / 3)

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the uneven torque disturbances."""

        if self._use_sinusoidal_torque:
            self._torque_x_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_y_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_z_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_x_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_y_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_z_offset = torch.zeros(
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
            self._torque_x_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._torque_y_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._torque_z_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._torque_x_offset[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_offset - self._min_offset)
                + self._min_offset
            )
            self._torque_y_offset[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_offset - self._min_offset)
                + self._min_offset
            )
            self._torque_z_offset[env_ids] = (
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
            theta = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * math.pi
                * 2
            )
            phi = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * math.pi
            )
            self.torque_forces[env_ids, 0] = torch.cos(theta) * r
            self.torque_forces[env_ids, 1] = torch.sin(theta) * r
            self.torque_forces[env_ids, 2] = torch.cos(phi) * r

    def get_torque_disturbance(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the torque forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The torque forces to apply to the robot."""

        if self._use_sinusoidal_torque:
            self.torque_forces[:, 0] = (
                torch.sin(root_pos[:, 0] * self._torque_x_freq + self._torque_x_offset)
                * self._max_torque
            )
            self.torque_forces[:, 1] = (
                torch.sin(root_pos[:, 1] * self._torque_y_freq + self._torque_y_offset)
                * self._max_torque
            )
            self.torque_forces[:, 2] = (
                torch.sin(root_pos[:, 2] * self._torque_z_freq + self._torque_z_offset)
                * self._max_torque
            )

        return self.torque_forces


class NoisyObservations(NoisyObservations2D):
    """
    Adds noise to the observations of the robot."""

    def __init__(self, task_cfg: dict) -> None:
        """
        Args:
            task_cfg (dict): The task configuration."""

        super(NoisyObservations, self).__init__(task_cfg)


class NoisyActions(NoisyActions2D):
    """
    Adds noise to the actions of the robot."""

    def __init__(self, task_cfg: dict) -> None:
        """
        Args:
            task_cfg (dict): The task configuration."""

        super(NoisyActions, self).__init__(task_cfg)
