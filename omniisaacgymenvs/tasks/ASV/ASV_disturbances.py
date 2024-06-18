__author__ = "Antoine Richard, Junghwan Ro, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Junghwan Ro"
__email__ = "jro37@gatech.edu"
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
        else:
            self.platforms_mass[env_ids, 0] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device) * 0
                + self._base_mass
            )

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
        # if self._add_mass_disturbances:
        if True:
            body.set_masses(self.platforms_mass[idx, 0], indices=idx)


class ForceDisturbance:
    """
    Creates force disturbance."""

    def __init__(self, task_cfg: dict, num_envs: int, device: str) -> None:
        """
        Args:
            task_cfg (dict): The task configuration.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored."""
        self._use_force_disturbance = task_cfg["use_force_disturbance"]
        self._use_constant_force = task_cfg["use_constant_force"]
        self._use_sinusoidal_force = task_cfg["use_sinusoidal_force"]

        self._const_min = task_cfg["force_const_min"]
        self._const_max = task_cfg["force_const_max"]
        self._const_min = math.sqrt(self._const_min**2 / 2)
        self._const_max = math.sqrt(self._const_max**2 / 2)

        self._sin_min = task_cfg["force_sin_min"]
        self._sin_max = task_cfg["force_sin_max"]
        self._sin_min = math.sqrt(self._sin_min**2 / 2)
        self._sin_max = math.sqrt(self._sin_max**2 / 2)
        self._min_freq = task_cfg["force_min_freq"]
        self._max_freq = task_cfg["force_max_freq"]
        self._min_shift = task_cfg["force_min_shift"]
        self._max_shift = task_cfg["force_max_shift"]

        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the force disturbances."""

        if self._use_sinusoidal_force:
            self._force_x_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_y_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_x_shift = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_y_shift = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._force_amp = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.disturbance_forces = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.disturbance_forces_const = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_force(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Generates the forces.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""
        if not self._use_force_disturbance:
            self.disturbance_forces[env_ids, 0] = 0
            self.disturbance_forces[env_ids, 1] = 0
            self.disturbance_forces[env_ids, 2] = 0
            return

        if self._use_sinusoidal_force:
            self._force_x_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._force_y_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._force_x_shift[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_shift - self._min_shift)
                + self._min_shift
            )
            self._force_y_shift[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_shift - self._min_shift)
                + self._min_shift
            )
            self._force_amp[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._sin_max - self._sin_min)
                + self._sin_min
            )
            # print(f"force_amp: {self._force_amp}")
        if self._use_constant_force:
            r = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * (self._const_max - self._const_min)
                + self._const_min
            )
            theta = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * math.pi
                * 2
            )
            # print(f"cos(theta)*r: {torch.cos(theta) * r}")
            # print(f"sin(theta)*r: {torch.sin(theta) * r}")
            self.disturbance_forces_const[env_ids, 0] = torch.cos(theta) * r
            self.disturbance_forces_const[env_ids, 1] = torch.sin(theta) * r
            # print(f"r: {r}")
            # print(f"theta: {theta}")
            # print(f"disturbance_forces_const: {self.disturbance_forces_const}")

    def get_disturbance_forces(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the disturbance forces for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The disturbance forces."""
        # print(f"disturbance_forces_const: {self.disturbance_forces_const}")
        if self._use_constant_force:
            self.disturbance_forces = self.disturbance_forces_const.clone()

        if self._use_sinusoidal_force:
            self.disturbance_forces[:, 0] = self.disturbance_forces_const[:, 0] + (
                torch.sin(root_pos[:, 0] * self._force_x_freq + self._force_x_shift)
                * self._force_amp
            )
            self.disturbance_forces[:, 1] = self.disturbance_forces_const[:, 1] + (
                torch.sin(root_pos[:, 1] * self._force_y_freq + self._force_y_shift)
                * self._force_amp
            )

        # print(f"disturbance_forces: {self.disturbance_forces}")
        return self.disturbance_forces


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

        # Disturbance torque generation
        self._use_torque_disturbance = task_cfg["use_torque_disturbance"]
        self._use_constant_torque = task_cfg["use_constant_torque"]
        self._use_sinusoidal_torque = task_cfg["use_sinusoidal_torque"]

        self._const_min = task_cfg["torque_const_min"]
        self._const_max = task_cfg["torque_const_max"]

        self._sin_min = task_cfg["torque_sin_min"]
        self._sin_max = task_cfg["torque_sin_max"]

        # use the same min/max frequencies and offsets for the force
        self._min_freq = task_cfg["torque_min_freq"]
        self._max_freq = task_cfg["torque_max_freq"]
        self._min_shift = task_cfg["torque_min_shift"]
        self._max_shift = task_cfg["torque_max_shift"]

        self._num_envs = num_envs
        self._device = device

        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the disturbances."""

        if self._use_sinusoidal_torque:
            self._torque_freq = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_shift = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )
            self._torque_amp = torch.zeros(
                (self._num_envs), device=self._device, dtype=torch.float32
            )

        self.disturbance_torques = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.disturbance_torques_const = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )

    def generate_torque(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Generates the torque disturbance.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            num_resets (int): The number of resets to perform."""

        if not self._use_torque_disturbance:
            self.disturbance_torques[env_ids, 2] = 0
            return

        if self._use_sinusoidal_torque:
            #  use the same min/max frequencies and offsets for the force
            self._torque_freq[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_freq - self._min_freq)
                + self._min_freq
            )
            self._torque_shift[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._max_shift - self._min_shift)
                + self._min_shift
            )
            self._torque_amp[env_ids] = (
                torch.rand(num_resets, dtype=torch.float32, device=self._device)
                * (self._sin_max - self._sin_min)
                + self._sin_min
            )
        if self._use_constant_torque:
            r = (
                torch.rand((num_resets), dtype=torch.float32, device=self._device)
                * (self._const_max - self._const_min)
                + self._const_min
            )
            # make torques negative for half of the environments at random
            r[
                torch.rand((num_resets), dtype=torch.float32, device=self._device) > 0.5
            ] *= -1
            self.disturbance_torques_const[env_ids, 2] = r

    def get_torque_disturbance(self, root_pos: torch.Tensor) -> torch.Tensor:
        """
        Computes the torques for the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The torque disturbance."""
        if self._use_constant_torque:
            self.disturbance_torques = self.disturbance_torques_const.clone()
        if self._use_sinusoidal_torque:
            self.disturbance_torques[:, 2] = self.disturbance_torques_const[:, 2] + (
                torch.sin(
                    (root_pos[:, 0] + root_pos[:, 1]) * self._torque_freq
                    + self._torque_shift
                )
                * self._torque_amp
            )

        return self.disturbance_torques


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
