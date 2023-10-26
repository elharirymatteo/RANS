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

        self._rng = np.random.default_rng(seed=cfg["seed"])
        self._num_thrusters_to_kill = cfg["num_thrusters_to_kill"]
        self.killed_thrusters_id = []

    def generate_thruster_kills(self) -> None:
        """
        Generates the thrusters to kill."""

        self.killed_thrusters_id = self._rng.choice(
            8, self._num_thrusters_to_kill, replace=False
        )  # [2,3]
        print("Killed thrusters: ", self.killed_thrusters_id)


class UnevenFloorDisturbance:
    """
    Creates disturbances on the platform by simulating an uneven floor."""

    def __init__(self, cfg: Dict[str, float]) -> None:
        """
        Initialize the uneven floor disturbance.

        Args:
            cfg (Dict[str,float]): A dictionary containing the configuration of the uneven floor disturbance.
        """

        self._rng = np.random.default_rng(seed=cfg["seed"])
        self._use_uneven_floor = cfg["use_uneven_floor"]
        self._use_sinusoidal_floor = cfg["use_sinusoidal_floor"]
        self._min_freq = cfg["floor_min_freq"]
        self._max_freq = cfg["floor_max_freq"]
        self._min_offset = cfg["floor_min_offset"]
        self._max_offset = cfg["floor_max_offset"]
        self._max_floor_force = cfg["max_floor_force"]
        self._min_floor_force = cfg["min_floor_force"]
        self._max_floor_force = math.sqrt(self._max_floor_force**2 / 2)
        self._min_floor_force = math.sqrt(self._min_floor_force**2 / 2)

        self._floor_forces = np.zeros(3, dtype=np.float32)
        self._floor_x_freq = 0
        self._floor_y_freq = 0
        self._floor_x_offset = 0
        self._floor_y_offset = 0

    def generate_floor(self) -> None:
        """
        Generates the uneven floor."""

        if self._use_uneven_floor:
            if self._use_sinusoidal_floor:
                self._floor_x_freq = self._rng.uniform(
                    self._min_freq, self._max_freq, 1
                )
                self._floor_y_freq = self._rng.uniform(
                    self._min_freq, self._max_freq, 1
                )
                self._floor_x_offset = self._rng.uniform(
                    self._min_offset, self._max_offset, 1
                )
                self._floor_y_offset = self._rng.uniform(
                    self._min_offset, self._max_offset, 1
                )
            else:
                r = self._rng.uniform(self._min_floor_force, self._max_floor_force, 1)
                theta = self._rng.uniform(0, 1, 1) * math.pi * 2
                self._floor_forces[0] = np.cos(theta) * r
                self._floor_forces[1] = np.sin(theta) * r

    def get_floor_forces(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the floor forces for the current state of the robot.

        Args:
            root_pos (np.ndarray): The position of the root of the robot.

        Returns:
            np.ndarray: The floor forces."""

        if self._use_uneven_floor:
            if self._use_sinusoidal_floor:
                self._floor_forces[0] = (
                    np.sin(root_pos[0] * self._floor_x_freq + self._floor_x_offset)
                    * self._max_floor_force
                )
                self._floor_forces[1] = (
                    np.sin(root_pos[1] * self._floor_y_freq + self._floor_y_offset)
                    * self._max_floor_force
                )

        return self._floor_forces


class TorqueDisturbance:
    """
    Creates disturbances on the platform by simulating a torque applied to its center.
    """

    def __init__(self, cfg: Dict[str, float]) -> None:
        """
        Initialize the torque disturbance.

        Args:
            cfg (Dict[str,float]): A dictionary containing the configuration of the torque disturbance.
        """

        self._rng = np.random.default_rng(seed=cfg["seed"])
        # Uneven floor generation
        self._use_torque_disturbance = cfg["use_torque_disturbance"]
        self._use_sinusoidal_torque = cfg["use_sinusoidal_torque"]
        self._max_torque = cfg["max_torque"]
        self._min_torque = cfg["min_torque"]

        # use the same min/max frequencies and offsets for the floor
        self._min_freq = cfg["floor_min_freq"]
        self._max_freq = cfg["floor_max_freq"]
        self._min_offset = cfg["floor_min_offset"]
        self._max_offset = cfg["floor_max_offset"]

        self._torque_forces = np.zeros(3, dtype=np.float32)
        self._torque_freq = 0
        self._torque_offset = 0

    def generate_torque(self) -> None:
        """
        Generates the torque disturbance."""

        if self._use_torque_disturbance:
            if self._use_sinusoidal_torque:
                #  use the same min/max frequencies and offsets for the floor
                self._torque_freq = self._rng.uniform(self._min_freq, self._max_freq, 1)
                self._torque_offset = self._rng.uniform(
                    self._min_offset, self._max_offset, 1
                )
            else:
                r = self._rng.uniform(
                    self._min_torque, self._max_torque, 1
                ) * self._rng.choice([1, -1])
                self._torque_forces[2] = r

    def get_torque_disturbance(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the torque for the current state of the robot.

        Args:
            root_pos (np.ndarray): The position of the root of the robot.

        Returns:
            np.ndarray: The torque."""

        if self._use_torque_disturbance:
            if self._use_sinusoidal_torque:
                self._torque_forces[2] = (
                    np.sin(root_pos * self._torque_freq + self._torque_offset)
                    * self._max_torque
                )

        return self._torque_forces


class NoisyObservations:
    """
    Adds noise to the observations of the robot."""

    def __init__(self, cfg: Dict[str, float]) -> None:
        """
        Initialize the noisy observations strategy.

        Args:
            cfg (Dict[str,float]): A dictionary containing the configuration of the noisy observations disturbance.
        """

        self._rng = np.random.default_rng(seed=cfg["seed"])
        self._add_noise_on_pos = cfg["add_noise_on_pos"]
        self._position_noise_min = cfg["position_noise_min"]
        self._position_noise_max = cfg["position_noise_max"]
        self._add_noise_on_vel = cfg["add_noise_on_vel"]
        self._velocity_noise_min = cfg["velocity_noise_min"]
        self._velocity_noise_max = cfg["velocity_noise_max"]
        self._add_noise_on_heading = cfg["add_noise_on_heading"]
        self._heading_noise_min = cfg["heading_noise_min"]
        self._heading_noise_max = cfg["heading_noise_max"]

    def add_noise_on_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Adds noise to the position of the robot.

        Args:
            pos (np.ndarray): The position of the robot.

        Returns:
            np.ndarray: The position of the robot with noise added."""

        if self._add_noise_on_pos:
            pos += self._rng.uniform(
                self._position_noise_min, self._position_noise_max, pos.shape
            )
        return pos

    def add_noise_on_vel(self, vel: np.ndarray) -> np.ndarray:
        """
        Adds noise to the velocity of the robot.

        Args:
            vel (np.ndarray): The velocity of the robot.

        Returns:
            np.ndarray: The velocity of the robot with noise added."""

        if self._add_noise_on_vel:
            vel += self._rng.uniform(
                self._velocity_noise_min, self._velocity_noise_max, vel.shape
            )
        return vel

    def add_noise_on_heading(self, heading: np.ndarray) -> np.ndarray:
        """
        Adds noise to the heading of the robot.

        Args:
            heading (np.ndarray): The heading of the robot.

        Returns:
            np.ndarray: The heading of the robot with noise added."""

        if self._add_noise_on_heading:
            heading += self._rng.uniform(
                self._heading_noise_min, self._heading_noise_max, heading.shape
            )
        return heading


class NoisyActions:
    """
    Adds noise to the actions of the robot."""

    def __init__(self, cfg: Dict[str, float]) -> None:
        """
        Initialize the noisy actions strategy.

        Args:
            cfg (Dict[str,float]): A dictionary containing the configuration of the noisy actions disturbance.
        """

        self._rng = np.random.default_rng(seed=cfg["seed"])
        self._add_noise_on_act = cfg["add_noise_on_act"]
        self._min_action_noise = cfg["min_action_noise"]
        self._max_action_noise = cfg["max_action_noise"]

    def add_noise_on_act(self, act: np.ndarray) -> np.ndarray:
        """
        Adds noise to the actions of the robot.

        Args:
            act (np.ndarray): The actions of the robot.

        Returns:
            np.ndarray: The actions of the robot with noise added."""

        if self._add_noise_on_act:
            act += self._rng.uniform(self._min_action_noise, self._max_action_noise, 1)
        return act
