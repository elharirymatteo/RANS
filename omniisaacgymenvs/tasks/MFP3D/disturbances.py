__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_3DoF.disturbances_parameters import (
    DisturbancesParameters,
    MassDistributionDisturbanceParameters,
    ForceDisturbanceParameters,
    TorqueDisturbanceParameters,
    NoisyObservationsParameters,
    NoisyActionsParameters,
)

from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    ForceDisturbance as ForceDisturbance2D,
)
from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    TorqueDisturbance as TorqueDisturbance2D,
)
from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    NoisyActions as NoisyActions2D,
)
from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    NoisyObservations as NoisyObservations2D,
)
from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    MassDistributionDisturbances as MassDistributionDisturbances2D,
)
from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    Disturbances as Disturbances2D,
)

from typing import Tuple
import torch
import math
import omni


class MassDistributionDisturbances(MassDistributionDisturbances2D):
    """
    Creates disturbances on the platform by simulating a mass distribution on the
    platform.
    """

    def __init__(
        self,
        parameters: MassDistributionDisturbanceParameters,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Args:
            parameters (MassDistributionDisturbanceParameters): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        super(MassDistributionDisturbances, self).__init__(parameters, num_envs, device)

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the mass disturbances.
        """

        super().instantiate_buffers()
        self.platforms_CoM = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def randomize_masses(self, env_ids: torch.Tensor, step: int = 0) -> None:
        """
        Randomizes the masses of the platforms.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            step (int): The current step of the learning process.
        """

        num_resets = len(env_ids)
        self.platforms_mass[env_ids, 0] = self.mass_sampler.sample(
            num_resets, step, device=self._device
        )
        r = self.CoM_sampler.sample(num_resets, step, device=self._device)
        theta = (
            torch.rand((num_resets), dtype=torch.float32, device=self._device)
            * math.pi
            * 2
        )
        phi = (
            torch.rand((num_resets), dtype=torch.float32, device=self._device) * math.pi
        )
        self.platforms_CoM[env_ids, 0] = torch.cos(theta) * torch.sin(phi) * r
        self.platforms_CoM[env_ids, 1] = torch.sin(theta) * torch.sin(phi) * r
        self.platforms_CoM[env_ids, 2] = torch.cos(phi) * r

    def set_coms(
        self,
        body: omni.isaac.core.prims.XFormPrimView,
        env_ids: torch.Tensor,
        joints_idx: Tuple[int, int],
    ) -> None:
        """
        Sets the CoM of the platforms.

        Args:
            body (omni.isaac.core.XFormPrimView): The rigid bodies containing the prismatic joints controlling the position of the CoMs.
            env_ids (torch.Tensor): The ids of the environments to reset.
            joints_idx (Tuple[int, int]): The ids of the x and y joints respectively.
        """

        joints_position = torch.zeros(
            (len(env_ids), 3), device=self._device, dtype=torch.float32
        )
        joints_position[:, joints_idx[0]] = self.platforms_CoM[env_ids, 0]
        joints_position[:, joints_idx[1]] = self.platforms_CoM[env_ids, 1]
        joints_position[:, joints_idx[2]] = self.platforms_CoM[env_ids, 2]
        if self.parameters.enable:
            body.set_joint_positions(joints_position, indices=env_ids)


class ForceDisturbance(ForceDisturbance2D):
    """
    Creates disturbances on the platform by simulating an uneven floor.
    """

    def __init__(
        self, parameters: ForceDisturbanceParameters, num_envs: int, device: str
    ) -> None:
        """
        Args:
            parameters (ForceDisturbanceParameters): The task configuration.
            num_envs (int): The number of environments to create.
            device (str): The device to use for the computation.
        """

        super(ForceDisturbance, self).__init__(parameters, num_envs, device)

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the uneven floor disturbances.
        """

        if self.parameters.use_sinusoidal_patterns:
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
            self._max_forces = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.forces = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_floor(
        self, env_ids: torch.Tensor, num_resets: int, step: int = 0
    ) -> None:
        """
        Generates the uneven floor.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform.
            step (int, optional): The current training step. Defaults to 0.
        """

        if self.parameters.enable:
            if self.parameters.use_sinusoidal_patterns:
                self._floor_x_freq[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_freq - self.parameters.min_freq)
                    + self.parameters.min_freq
                )
                self._floor_y_freq[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_freq - self.parameters.min_freq)
                    + self.parameters.min_freq
                )
                self._floor_z_freq[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_freq - self.parameters.min_freq)
                    + self.parameters.min_freq
                )
                self._floor_x_offset[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_offset - self.parameters.min_offset)
                    + self.parameters.min_offset
                )
                self._floor_y_offset[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_offset - self.parameters.min_offset)
                    + self.parameters.min_offset
                )
                self._floor_z_offset[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_offset - self.parameters.min_offset)
                    + self.parameters.min_offset
                )
                self._max_forces[env_ids] = self.force_sampler.sample(
                    num_resets, step, device=self._device
                )
            else:
                r = self.force_sampler.sample(num_resets, step, device=self._device)
                theta = (
                    torch.rand((num_resets), dtype=torch.float32, device=self._device)
                    * math.pi
                    * 2
                )
                phi = (
                    torch.rand((num_resets), dtype=torch.float32, device=self._device)
                    * math.pi
                )
                self.forces[env_ids, 0] = torch.cos(theta) * torch.sin(phi) * r
                self.forces[env_ids, 1] = torch.sin(theta) * torch.sin(phi) * r
                self.forces[env_ids, 2] = torch.cos(phi) * r

    def get_floor_forces(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the floor forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The floor forces to apply to the robot.
        """

        if self.parameters.use_sinusoidal_patterns:
            self.forces[:, 0] = (
                torch.sin(root_pos[:, 0] * self._floor_x_freq + self._floor_x_offset)
                * self._max_forces
            )
            self.forces[:, 1] = (
                torch.sin(root_pos[:, 1] * self._floor_y_freq + self._floor_y_offset)
                * self._max_forces
            )
            self.forces[:, 2] = (
                torch.sin(root_pos[:, 2] * self._floor_z_freq + self._floor_z_offset)
                * self._max_forces
            )

        return self.forces


class TorqueDisturbance(TorqueDisturbance2D):
    """
    Creates disturbances on the platform by simulating a torque applied to its center.
    """

    def __init__(
        self, parameters: TorqueDisturbanceParameters, num_envs: int, device: str
    ) -> None:
        """
        Args:
            parameters (TorqueDisturbanceParameters): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        super(TorqueDisturbance, self).__init__(parameters, num_envs, device)

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the uneven torque disturbances."""

        if self.parameters.use_sinusoidal_patterns:
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
            self._max_torques = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.torques = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_torque(
        self, env_ids: torch.Tensor, num_resets: int, step: int = 0
    ) -> None:
        """
        Generates the torque disturbance.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform.
            step (int, optional): The current training step. Defaults to 0.
        """

        if self.parameters.enable:
            if self.parameters.use_sinusoidal_patterns:
                #  use the same min/max frequencies and offsets for the floor
                self._torque_x_freq[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_freq - self.parameters.min_freq)
                    + self.parameters.min_freq
                )
                self._torque_y_freq[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_freq - self.parameters.min_freq)
                    + self.parameters.min_freq
                )
                self._torque_z_freq[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_freq - self.parameters.min_freq)
                    + self.parameters.min_freq
                )
                self._torque_x_offset[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_offset - self.parameters.min_offset)
                    + self.parameters.min_offset
                )
                self._torque_y_offset[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_offset - self.parameters.min_offset)
                    + self.parameters.min_offset
                )
                self._torque_z_offset[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_offset - self.parameters.min_offset)
                    + self.parameters.min_offset
                )
                self._max_torques[env_ids] = self.torque_sampler.sample(
                    num_resets, step, device=self._device
                )
            else:
                r = self.torque_sampler.sample(num_resets, step, device=self._device)
                theta = (
                    torch.rand((num_resets), dtype=torch.float32, device=self._device)
                    * math.pi
                    * 2
                )
                phi = (
                    torch.rand((num_resets), dtype=torch.float32, device=self._device)
                    * math.pi
                )
                self.torques[env_ids, 0] = torch.cos(theta) * torch.sin(phi) * r
                self.torques[env_ids, 1] = torch.sin(theta) * torch.sin(phi) * r
                self.torques[env_ids, 2] = torch.cos(phi) * r

    def get_torque_disturbance(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the torque forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The torque forces to apply to the robot.
        """

        if self.parameters.use_sinusoidal_patterns:
            self.torques[:, 0] = (
                torch.sin(root_pos[:, 0] * self._torque_x_freq + self._torque_x_offset)
                * self._max_torques
            )
            self.torques[:, 1] = (
                torch.sin(root_pos[:, 1] * self._torque_y_freq + self._torque_y_offset)
                * self._max_torques
            )
            self.torques[:, 2] = (
                torch.sin(root_pos[:, 2] * self._torque_z_freq + self._torque_z_offset)
                * self._max_torques
            )

        return self.torques


class NoisyObservations(NoisyObservations2D):
    """
    Adds noise to the observations of the robot.
    """

    def __init__(
        self, parameters: NoisyObservationsParameters, num_envs: int, device: str
    ) -> None:
        """
        Args:
            task_cfg (NoisyObservationParameters): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        super(NoisyObservations, self).__init__(parameters, num_envs, device)


class NoisyActions(NoisyActions2D):
    """
    Adds noise to the actions of the robot.
    """

    def __init__(
        self, parameters: NoisyActionsParameters, num_envs: int, device: str
    ) -> None:
        """
        Args:
            parameters (NoisyActionParameters): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        super(NoisyActions, self).__init__(parameters, num_envs, device)


class Disturbances(Disturbances2D):
    """
    Class to create disturbances on the platform.
    """

    def __init__(
        self,
        parameters: dict,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Args:
            parameters (dict): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self._num_envs = num_envs
        self._device = device

        self.parameters = DisturbancesParameters(**parameters)

        self.mass_disturbances = MassDistributionDisturbances(
            self.parameters.mass_disturbance,
            num_envs,
            device,
        )
        self.force_disturbances = ForceDisturbance(
            self.parameters.force_disturbance,
            num_envs,
            device,
        )
        self.torque_disturbances = TorqueDisturbance(
            self.parameters.torque_disturbance,
            num_envs,
            device,
        )
        self.noisy_observations = NoisyObservations(
            self.parameters.observations_disturbance,
            num_envs,
            device,
        )
        self.noisy_actions = NoisyActions(
            self.parameters.actions_disturbance,
            num_envs,
            device,
        )
