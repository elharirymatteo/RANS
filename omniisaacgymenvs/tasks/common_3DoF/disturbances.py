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
    NoisyImagesParameters,
)

from omniisaacgymenvs.tasks.common_3DoF.curriculum_helpers import (
    CurriculumSampler,
)

from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import wandb
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

        if self.parameters.enable:
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

    def get_masses(
        self,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns the masses and CoM of the platforms.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): The masses and CoM of the platforms.
        """

        return self.platforms_mass[env_ids, 0]

    def get_masses_and_com(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the masses and CoM of the platforms.

        Returns:
            Tuple(torch.Tensor, torch.Tensor): The masses and CoM of the platforms.
        """

        return torch.cat((self.platforms_mass, self.platforms_CoM), axis=1)

    def get_CoM(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns the CoM of the platforms.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.

        Returns:
            torch.Tensor: The CoM of the platforms.
        """

        return self.platforms_CoM[env_ids]

    def get_image_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            mass = self.platforms_mass.cpu().numpy().flatten()
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(mass, bins=32)
            ax.set_title("Mass disturbance")
            ax.set_xlim(
                self.mass_sampler.get_min_bound(), self.mass_sampler.get_max_bound()
            )
            ax.set_xlabel("mass (Kg)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/mass_disturbance"] = wandb.Image(data)
        if self.parameters.enable:
            com = torch.norm(self.platforms_CoM.cpu(), axis=-1).numpy().flatten()
            fig, ax = plt.subplots(1, 2, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(com, bins=32)
            ax.set_title("CoM disturbance")
            ax.set_xlim(
                self.CoM_sampler.get_min_bound(), self.CoM_sampler.get_max_bound()
            )
            ax.set_xlabel("Displacement (m)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/CoM_disturbance"] = wandb.Image(data)
        return dict

    def get_scalar_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            dict["disturbance/mass_disturbance_rate"] = self.mass_sampler.get_rate(step)
            dict["disturbance/CoM_disturbance_rate"] = self.CoM_sampler.get_rate(step)
        return dict


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

    def get_force_disturbance(self, root_pos: torch.Tensor) -> torch.Tensor:
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

    def get_image_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            force = self.force_sampler.sample(self._num_envs, step, device=self._device)
            force = force.cpu().numpy().flatten()
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(force, bins=32)
            ax.set_title("Force disturbance")
            ax.set_xlim(0, self.force_sampler.get_max_bound())
            ax.set_xlabel("force (N)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/force_disturbance"] = wandb.Image(data)
        return dict

    def get_scalar_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            dict["disturbance/force_disturbance_rate"] = self.force_sampler.get_rate(
                step
            )
        return dict


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

    def get_image_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            torque = self.torque_sampler.sample(
                self._num_envs, step, device=self._device
            )
            torque = torque.cpu().numpy().flatten()
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(torque, bins=32)
            ax.set_title("Torque disturbance")
            ax.set_xlim(
                self.torque_sampler.get_min_bound(), self.torque_sampler.get_max_bound()
            )
            ax.set_xlabel("torque (Nm)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/torque_disturbance"] = wandb.Image(data)
        return dict

    def get_scalar_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            dict["disturbance/torque_disturbance_rate"] = self.torque_sampler.get_rate(
                step
            )
        return dict


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
            self.pos_shape = pos.shape
            pos += self.position_sampler.sample(
                self._num_envs * pos.shape[1], step, device=self._device
            ).reshape(-1, self.pos_shape[1])
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
            self.vel_shape = vel.shape
            vel += self.velocity_sampler.sample(
                self._num_envs * vel.shape[1], step, device=self._device
            ).reshape(-1, self.vel_shape[1])
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

    def get_image_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """

        dict = {}

        if self.parameters.enable_position_noise:
            position = self.position_sampler.sample(
                self._num_envs * self.pos_shape[1], step, device=self._device
            )
            position = position.cpu().numpy().flatten()
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(position, bins=32)
            ax.set_title("Position noise")
            ax.set_xlim(
                self.position_sampler.get_min_bound(),
                self.position_sampler.get_max_bound(),
            )
            ax.set_xlabel("noise (m)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/position_noise"] = wandb.Image(data)
        if self.parameters.enable_velocity_noise:
            velocity = self.velocity_sampler.sample(
                self._num_envs * self.vel_shape[1], step, device=self._device
            )
            velocity = velocity.cpu().numpy().flatten()
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(velocity, bins=32)
            ax.set_title("Velocity noise")
            ax.set_xlim(
                self.velocity_sampler.get_min_bound(),
                self.position_sampler.get_max_bound(),
            )
            ax.set_xlabel("noise (m/s)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/velocity_noise"] = wandb.Image(data)
        if self.parameters.enable_orientation_noise:
            orientation = self.orientation_sampler.sample(
                self._num_envs, step, device=self._device
            )
            orientation = orientation.cpu().numpy().flatten()
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(orientation, bins=32)
            ax.set_title("Orientation noise")
            ax.set_xlim(
                self.orientation_sampler.get_min_bound(),
                self.orientation_sampler.get_max_bound(),
            )
            ax.set_xlabel("noise (rad)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/orientation_noise"] = wandb.Image(data)
        return dict

    def get_scalar_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable_position_noise:
            dict["disturbance/position_disturbance_rate"] = (
                self.position_sampler.get_rate(step)
            )
        if self.parameters.enable_velocity_noise:
            dict["disturbance/velocity_disturbance_rate"] = (
                self.velocity_sampler.get_rate(step)
            )
        if self.parameters.enable_orientation_noise:
            dict["disturbance/orientation_disturbance_rate"] = (
                self.orientation_sampler.get_rate(step)
            )
        return dict


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
            self.shape = act.shape
            act += self.action_sampler.sample(
                self._num_envs * act.shape[1], step, device=self._device
            ).reshape(-1, self.shape[1])
        return act

    def get_image_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            action = self.action_sampler.sample(
                self._num_envs * self.shape[1], step, device=self._device
            ).reshape(-1, self.shape[1])
            action = action.cpu().numpy().flatten()
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.hist(action, bins=32)
            ax.set_title("Action noise")
            ax.set_xlim(
                self.action_sampler.get_min_bound(), self.action_sampler.get_max_bound()
            )
            ax.set_xlabel("noise (N)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict["disturbance/action_noise"] = wandb.Image(data)
        return dict

    def get_scalar_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            dict["disturbance/action_disturbance_rate"] = self.action_sampler.get_rate(
                step
            )
        return dict


class NoisyImages:
    """
    Adds noise to the actions of the robot."""

    def __init__(
        self,
        parameters: NoisyImagesParameters,
        num_envs: int,
        device: str,
    ) -> None:
        """
        Args:
            parameters (NoisyActionParameters): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self.image_sampler = CurriculumSampler(parameters.image_curriculum)
        self.parameters = parameters
        self._num_envs = num_envs
        self._device = device

    def add_noise_on_image(self, image: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Adds noise to the actions of the robot.

        Args:
            image (torch.Tensor): The image observation of the robot. Shape is (num_envs, channel, height, width).
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            torch.Tensor: The image observation of the robot with noise.
        """

        if self.parameters.enable:
            self.shape = image.shape
            image += self.image_sampler.sample(
                self._num_envs * self.shape[1] * self.shape[2] * self.shape[3],
                step,
                device=self._device,
            ).reshape(-1, self.shape[1], self.shape[2], self.shape[3])
        return image

    def get_image_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            image = self.image_sampler.sample(
                self._num_envs * self.shape[1] * self.shape[2] * self.shape[3],
                step,
                device=self._device,
            ).reshape(-1, self.shape[1], self.shape[2], self.shape[3])
            image = image.squeeze().cpu().numpy()[0]
            fig, ax = plt.subplots(1, 1, dpi=100, figsize=(8, 8), sharey=True)
            ax.imshow(image)
            ax.set_title("Action noise")
            fig.tight_layout()
            fig.canvas.draw()
            data = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close(fig)
            dict[f"disturbance/{self.parameters.modality}_noise"] = wandb.Image(data)
        return dict

    def get_scalar_logs(self, step: int) -> dict:
        """
        Logs the current state of the disturbances.

        Args:
            step (int): The current step of the learning process.

        Returns:
            dict: The logged data.
        """
        dict = {}

        if self.parameters.enable:
            dict[f"disturbance/{self.parameters.modality}_disturbance_rate"] = (
                self.image_sampler.get_rate(step)
            )
        return dict


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
        self.noisy_rgb_images = NoisyImages(
            self.parameters.rgb_disturbance, num_envs, device
        )
        self.noisy_depth_images = NoisyImages(
            self.parameters.depth_disturbance, num_envs, device
        )

    def get_logs(self, step: int) -> dict:
        """
        Collects logs for all the disturbances.

        Args:
            step (int): The current training step.

        Returns:
            dict: The logs for all used disturbances.
        """
        dict = {}
        dict = {**dict, **self.mass_disturbances.get_scalar_logs(step)}
        dict = {**dict, **self.force_disturbances.get_scalar_logs(step)}
        dict = {**dict, **self.torque_disturbances.get_scalar_logs(step)}
        dict = {**dict, **self.noisy_observations.get_scalar_logs(step)}
        dict = {**dict, **self.noisy_actions.get_scalar_logs(step)}
        if step % 50 == 0:
            dict = {**dict, **self.mass_disturbances.get_image_logs(step)}
            dict = {**dict, **self.force_disturbances.get_image_logs(step)}
            dict = {**dict, **self.torque_disturbances.get_image_logs(step)}
            dict = {**dict, **self.noisy_observations.get_image_logs(step)}
            dict = {**dict, **self.noisy_actions.get_image_logs(step)}
        return dict
