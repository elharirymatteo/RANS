__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_disturbances_parameters import (
    DisturbancesParameters,
    MassDistributionDisturbanceParameters,
    ForceDisturbanceParameters,
    TorqueDisturbanceParameters,
    NoisyObservationsParameters,
    NoisyActionsParameters,
)

from omniisaacgymenvs.tasks.virtual_floating_platform.curriculum_helpers import (
    CurriculumSampler,
)

from typing import Tuple
import torch
import math
import omni


class MassDistributionDisturbances:
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

        self.mass_sampler = CurriculumSampler(parameters.mass_curriculum)
        self.CoM_sampler = CurriculumSampler(parameters.com_curriculum)
        self.parameters = parameters
        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the mass disturbances.
        """

        self.platforms_mass = (
            torch.ones((self._num_envs, 1), device=self._device, dtype=torch.float32)
            * self.mass_sampler.get_min()
        )
        self.platforms_CoM = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
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
        self.platforms_CoM[env_ids, 0] = torch.cos(theta) * r
        self.platforms_CoM[env_ids, 1] = torch.sin(theta) * r

    def get_masses(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the masses and CoM of the platforms.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): The masses and CoM of the platforms.
        """

        return torch.cat((self.platforms_mass, self.platforms_CoM), axis=1)

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
            (len(env_ids), 2), device=self._device, dtype=torch.float32
        )
        joints_position[:, joints_idx[0]] = self.platforms_CoM[env_ids, 0]
        joints_position[:, joints_idx[1]] = self.platforms_CoM[env_ids, 1]
        if self.parameters.enable:
            body.set_joint_positions(joints_position, indices=env_ids)

    def set_masses(
        self,
        articulation_body: omni.isaac.core.prims.XFormPrimView,
        mass_body: omni.isaac.core.prims.XFormPrimView,
        env_ids: torch.Tensor,
        joints_idx: Tuple[int, int],
    ) -> None:
        """
        Sets the masses and CoM of the platforms.

        Args:
            articulation_body (omni.isaac.core.XFormPrimView): The rigid bodies containing the prismatic joints controlling the position of the CoMs.
            mass_body (omni.isaac.core.XFormPrimView): The rigid bodies containing the movable mass.
            env_ids(torch.Tensor): The ids of the environments to reset.
            joints_idx (Tuple[int, int]): The ids of the x and y joints respectively.
        """

        if self.parameters.enable:
            mass_body.set_masses(self.platforms_mass[env_ids, 0], indices=env_ids)
        self.set_coms(articulation_body, env_ids, joints_idx)


class ForceDisturbance:
    """
    Creates disturbances by applying random forces.
    """

    def __init__(
        self,
        parameters: ForceDisturbanceParameters,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Args:
            parameters (ForceDisturbanceParameters): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self.parameters = parameters
        self.force_sampler = CurriculumSampler(self.parameters.force_curriculum)
        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the force disturbances.
        """

        if self.parameters.use_sinusoidal_patterns:
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
            self._max_forces = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.forces = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_forces(
        self, env_ids: torch.Tensor, num_resets: int, step: int = 0
    ) -> None:
        """
        Generates the forces using a sinusoidal pattern or not.

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
                self.forces[env_ids, 0] = torch.cos(theta) * r
                self.forces[env_ids, 1] = torch.sin(theta) * r

    def get_floor_forces(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the forces given the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The floor forces.
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

        return self.forces


class TorqueDisturbance:
    """
    Creates disturbances by applying a torque to its center.
    """

    def __init__(
        self,
        parameters: TorqueDisturbanceParameters,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Args:
            parameters (TorqueDisturbanceParameters): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self.parameters = parameters
        self.torque_sampler = CurriculumSampler(self.parameters.torque_curriculum)
        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the torque disturbances.
        """

        if self.parameters.use_sinusoidal_patterns:
            self._torque_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_offset = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._max_torques = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.torques = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_torques(
        self, env_ids: torch.Tensor, num_resets: int, step: int = 0
    ) -> None:
        """
        Generates the torque disturbance.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform.
            step (int, optional): The current step of the training. Default to 0.
        """

        if self.parameters.enable:
            if self.parameters.use_sinusoidal_patterns:
                #  use the same min/max frequencies and offsets for the floor
                self._torque_freq[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_freq - self.parameters.min_freq)
                    + self.parameters.min_freq
                )
                self._torque_offset[env_ids] = (
                    torch.rand(num_resets, dtype=torch.float32, device=self._device)
                    * (self.parameters.max_offset - self.parameters.min_offset)
                    + self.parameters.min_offset
                )
                self._max_torques[env_ids] = self.torque_sampler.sample(
                    num_resets, step, device=self._device
                )
            else:
                self.torques[env_ids, 2] = self.torque_sampler.sample(
                    num_resets, step, device=self._device
                )

    def get_torque_disturbance(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the torques given the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The torque disturbance."""

        if self.parameters.use_sinusoidal_patterns:
            self.torques[:, 2] = (
                torch.sin(root_pos * self._torque_freq + self._torque_offset)
                * self._max_torques
            )

        return self.torques


class NoisyObservations:
    """
    Adds noise to the observations of the robot.
    """

    def __init__(
        self,
        parameters: NoisyObservationsParameters,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Args:
            task_cfg (NoisyObservationParameters): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self.position_sampler = CurriculumSampler(parameters.position_curriculum)
        self.velocity_sampler = CurriculumSampler(parameters.velocity_curriculum)
        self.orientation_sampler = CurriculumSampler(parameters.orientation_curriculum)
        self.parameters = parameters
        self._num_envs = num_envs
        self._device = device

    def add_noise_on_pos(self, pos: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Adds noise to the position of the robot.

        Args:
            pos (torch.Tensor): The position of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            torch.Tensor: The position of the robot with noise.
        """

        if self.parameters.enable_position_noise:
            pos += self.position_sampler.sample(
                self._num_envs * pos.shape[1], step, device=self._device
            ).reshape(-1, pos.shape[1])
        return pos

    def add_noise_on_vel(self, vel: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Adds noise to the velocity of the robot.

        Args:
            vel (torch.Tensor): The velocity of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            torch.Tensor: The velocity of the robot with noise.
        """

        if self.parameters.enable_velocity_noise:
            vel += self.velocity_sampler.sample(
                self._num_envs * vel.shape[1], step, device=self._device
            ).reshape(-1, vel.shape[1])
        return vel

    def add_noise_on_heading(
        self, heading: torch.Tensor, step: int = 0
    ) -> torch.Tensor:
        """
        Adds noise to the heading of the robot.

        Args:
            heading (torch.Tensor): The heading of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            torch.Tensor: The heading of the robot with noise.
        """

        if self.parameters.enable_orientation_noise:
            heading += self.orientation_sampler.sample(
                self._num_envs, step, device=self._device
            )
        return heading


class NoisyActions:
    """
    Adds noise to the actions of the robot."""

    def __init__(
        self,
        parameters: NoisyActionsParameters,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Args:
            parameters (NoisyActionParameters): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self.action_sampler = CurriculumSampler(parameters.action_curriculum)
        self.parameters = parameters
        self._num_envs = num_envs
        self._device = device

    def add_noise_on_act(self, act: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Adds noise to the actions of the robot.

        Args:
            act (torch.Tensor): The actions of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            torch.Tensor: The actions of the robot with noise.
        """

        if self.parameters.enable:
            act += self.action_sampler.sample(
                self._num_envs * act.shape[1], step, device=self._device
            ).reshape(-1, act.shape[1])
        return act


class Disturbances:
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
