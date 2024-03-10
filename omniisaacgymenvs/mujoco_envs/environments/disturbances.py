__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from typing import Dict, Tuple
import numpy as np
import math

from omniisaacgymenvs.tasks.MFP.MFP2D_disturbances_parameters import (
    DisturbancesParameters,
    MassDistributionDisturbanceParameters,
    ForceDisturbanceParameters,
    TorqueDisturbanceParameters,
    NoisyObservationsParameters,
    NoisyActionsParameters,
)

from omniisaacgymenvs.tasks.MFP.curriculum_helpers import (
    CurriculumSampler,
)


class RandomSpawn:
    """
    Randomly spawns the robot in the environment."""

    def __init__(self, cfg: Dict[str, float]) -> None:
        """
        Initialize the random spawn strategy.

        Args:
            cfg (dict): A dictionary containing the configuration of the random spawn disturbance.
        """

        self._rng = np.random.default_rng(seed=cfg["seed"])
        self._max_spawn_dist = cfg["max_spawn_dist"]
        self._min_spawn_dist = cfg["min_spawn_dist"]
        self._kill_dist = cfg["kill_dist"]

    def getInitialCondition(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a random initial condition for the robot.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the initial position and orientation of the robot.
        """

        theta = self._rng.uniform(-np.pi, np.pi, 1)
        r = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)
        initial_position = [np.cos(theta) * r, np.sin(theta) * r]
        heading = self._rng.uniform(-np.pi, np.pi, 1)
        initial_orientation = [np.cos(heading * 0.5), 0, 0, np.sin(heading * 0.5)]
        return initial_position, initial_orientation


class RandomKillThrusters:
    """
    Randomly kills thrusters."""

    def __init__(self, cfg: Dict[str, float]) -> None:
        """
        Initialize the random kill thrusters strategy.

        Args:
            cfg (dict): A dictionary containing the configuration of the random kill thrusters disturbance.
        """

        self._rng = np.random.default_rng(seed=42)  # cfg["seed"])
        self._num_thrusters_to_kill = cfg["num_thrusters_to_kill"]
        self.killed_thrusters_id = []
        self.killed_mask = np.ones([8])

    def generate_thruster_kills(self) -> None:
        """
        Generates the thrusters to kill."""

        self.killed_thrusters_id = self._rng.choice(
            8, self._num_thrusters_to_kill, replace=False
        )


class MassDistributionDisturbances:
    """
    Creates disturbances on the platform by simulating a mass distribution on the
    platform.
    """

    def __init__(
        self,
        rng: np.random.default_rng,
        parameters: MassDistributionDisturbanceParameters,
    ) -> None:
        """
        Args:
            parameters (MassDistributionDisturbanceParameters): The settings of the domain randomization.
        """

        self.rng = rng
        self.mass_sampler = CurriculumSampler(parameters.mass_curriculum)
        self.CoM_sampler = CurriculumSampler(parameters.com_curriculum)
        self.parameters = parameters
        self.platforms_mass = 5.32
        self.platforms_CoM = np.zeros((2), dtype=np.float32)

    def randomize_masses(self, step: int = 100000) -> None:
        """
        Randomizes the masses of the platforms.

        Args:
            env_ids (torch.Tensor): The ids of the environments to reset.
            step (int): The current step of the learning process.
        """

        self.platforms_mass = self.mass_sampler.sample(1, step).numpy()[0]
        r = self.CoM_sampler.sample(1, step).numpy()[0]
        theta = self.rand.uniform((1), dtype=np.float32) * math.pi * 2
        self.platforms_CoM[0] = np.cos(theta) * r
        self.platforms_CoM[1] = np.sin(theta) * r

    def get_masses(self) -> Tuple[float, np.ndarray]:
        """
        Returns the masses and CoM of the platforms.

        Returns:
            Tuple(float, np.ndarray): The masses and CoM of the platforms.
        """

        return (self.platforms_mass, self.platforms_CoM)


class ForceDisturbance:
    """
    Creates disturbances by applying random forces.
    """

    def __init__(
        self,
        rng: np.random.default_rng,
        parameters: ForceDisturbanceParameters,
    ) -> None:
        """
        Args:
            parameters (ForceDisturbanceParameters): The settings of the domain randomization.
        """

        self.rng = rng
        self.parameters = parameters
        self.force_sampler = CurriculumSampler(self.parameters.force_curriculum)

        self.forces = np.zeros(3, dtype=np.float32)
        self.max_forces = 0
        self._floor_x_freq = 0
        self._floor_y_freq = 0
        self._floor_x_offset = 0
        self._floor_y_offset = 0

    def generate_forces(self, step: int = 100000) -> None:
        """
        Generates the forces using a sinusoidal pattern or not.

        Args:
            step (int, optional): The current training step. Defaults to 0.
        """

        if self.parameters.enable:
            if self.parameters.use_sinusoidal_patterns:
                self._floor_x_freq = self.rng.uniform(
                    self.parameters.min_freq, self.parameters.max_freq, 1
                )
                self._floor_y_freq = self.rng.uniform(
                    self.parameters.min_freq, self.parameters.max_freq, 1
                )
                self._floor_x_offset = self.rng.uniform(
                    self.parameters.min_offset, self.parameters.max_offset, 1
                )
                self._floor_y_offset = self.rng.uniform(
                    self.parameters.min_offset, self.parameters.max_offset, 1
                )
                self._max_forces = self.force_sampler.sample(1, step).numpy()[0]
            else:
                r = self.force_sampler.sample(1, step).numpy()[0]
                theta = self.rng.uniform(0, 1, 1) * math.pi * 2
                self.forces[0] = np.cos(theta) * r
                self.forces[1] = np.sin(theta) * r

    def get_floor_forces(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the forces given the current state of the robot.

        Args:
            root_pos (np.ndarray): The position of the root of the robot.

        Returns:
            np.ndarray: The floor forces.
        """

        if self.parameters.use_sinusoidal_patterns:
            self.forces[0] = (
                np.sin(root_pos[0] * self._floor_x_freq + self._floor_x_offset)
                * self._max_forces
            )
            self.forces[1] = (
                np.sin(root_pos[1] * self._floor_y_freq + self._floor_y_offset)
                * self._max_forces
            )

        return self.forces


class TorqueDisturbance:
    """
    Creates disturbances by applying a torque to its center.
    """

    def __init__(
        self,
        rng: np.random.default_rng,
        parameters: TorqueDisturbanceParameters,
    ) -> None:
        """
        Args:
            parameters (TorqueDisturbanceParameters): The settings of the domain randomization.
        """

        self.rng = rng
        self.parameters = parameters
        self.torque_sampler = CurriculumSampler(self.parameters.torque_curriculum)

        self.torques = np.zeros(3, dtype=np.float32)
        self.max_torques = 0
        self._freq = 0
        self._offset = 0

    def generate_torques(self, step: int = 100000) -> None:
        """
        Generates the torques using a sinusoidal pattern or not.

        Args:
            step (int, optional): The current training step. Defaults to 0.
        """

        if self.parameters.enable:
            if self.parameters.use_sinusoidal_patterns:
                self._floor_x_freq = self.rng.uniform(
                    self.parameters.min_freq, self.parameters.max_freq, 1
                )
                self._floor_x_offset = self.rng.uniform(
                    self.parameters.min_offset, self.parameters.max_offset, 1
                )
                self._max_torques = self.torque_sampler.sample(1, step).numpy()[0]
            else:
                r = self.torque_sampler.sample(1, step).numpy()[0]
                self.torques[2] = r

    def get_torque_disturbance(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the torques given the current state of the robot.

        Args:
            root_pos (torch.Tensor): The position of the root of the robot.

        Returns:
            torch.Tensor: The torque disturbance."""

        if self.parameters.use_sinusoidal_patterns:
            self.torques[:, 2] = (
                np.sin(root_pos * self._freq + self._offset) * self._max_torques
            )

        return self.torques


class NoisyObservations:
    """
    Adds noise to the observations of the robot.
    """

    def __init__(
        self,
        rng: np.random.default_rng,
        parameters: NoisyObservationsParameters,
    ) -> None:
        """
        Args:
            task_cfg (NoisyObservationParameters): The settings of the domain randomization.
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self.rng = rng
        self.position_sampler = CurriculumSampler(parameters.position_curriculum)
        self.velocity_sampler = CurriculumSampler(parameters.velocity_curriculum)
        self.orientation_sampler = CurriculumSampler(parameters.orientation_curriculum)
        self.parameters = parameters

    def add_noise_on_pos(self, pos: np.ndarray, step: int = 100000) -> np.ndarray:
        """
        Adds noise to the position of the robot.

        Args:
            pos (np.ndarray): The position of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            np.ndarray: The position of the robot with noise.
        """

        if self.parameters.enable_position_noise:
            pos += self.position_sampler.sample(1, step).numpy()[0]
        return pos

    def add_noise_on_vel(self, vel: np.ndarray, step: int = 100000) -> np.ndarray:
        """
        Adds noise to the velocity of the robot.

        Args:
            vel (np.ndarray): The velocity of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            np.ndarray: The velocity of the robot with noise.
        """

        if self.parameters.enable_velocity_noise:
            vel += self.velocity_sampler.sample(1, step).numpy()[0]
        return vel

    def add_noise_on_heading(self, heading: np.ndarray, step: int = 0) -> np.ndarray:
        """
        Adds noise to the heading of the robot.

        Args:
            heading (np.ndarray): The heading of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            np.ndarray: The heading of the robot with noise.
        """

        if self.parameters.enable_orientation_noise:
            heading += self.orientation_sampler.sample(1, step).numpy()[0]
        return heading


class NoisyActions:
    """
    Adds noise to the actions of the robot."""

    def __init__(
        self,
        rng: np.random.default_rng,
        parameters: NoisyActionsParameters,
    ) -> None:
        """
        Args:
            parameters (NoisyActionParameters): The task configuration.
        """

        self.rng = rng
        self.action_sampler = CurriculumSampler(parameters.action_curriculum)
        self.parameters = parameters

    def add_noise_on_act(self, act: np.ndarray, step: int = 100000) -> np.ndarray:
        """
        Adds noise to the actions of the robot.

        Args:
            act (np.ndarray): The actions of the robot.
            step (int, optional): The current step of the learning process. Defaults to 0.

        Returns:
            np.ndarray: The actions of the robot with noise.
        """

        if self.parameters.enable:
            act += self.action_sampler.sample(1, step).numpy()[0]
        return act


class Disturbances:
    """
    Class to create disturbances on the platform.
    """

    def __init__(self, parameters: dict, seed: int = 42) -> None:
        """
        Args:
            parameters (dict): The settings of the domain randomization.
        """

        self.rng = np.random.default_rng(seed=seed)

        self.parameters = DisturbancesParameters(**parameters)

        self.mass_disturbances = MassDistributionDisturbances(
            self.rng,
            self.parameters.mass_disturbance,
        )
        self.force_disturbances = ForceDisturbance(
            self.rng,
            self.parameters.force_disturbance,
        )
        self.torque_disturbances = TorqueDisturbance(
            self.rng,
            self.parameters.torque_disturbance,
        )
        self.noisy_observations = NoisyObservations(
            self.rng,
            self.parameters.observations_disturbance,
        )
        self.noisy_actions = NoisyActions(
            self.rng,
            self.parameters.actions_disturbance,
        )
