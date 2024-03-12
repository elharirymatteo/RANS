__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from typing import Tuple
from pxr import Usd
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class Core:
    """
    The base class that implements the core of the task.
    """

    def __init__(self, num_envs: int, device: str) -> None:
        """
        The base class for the different subtasks.

        Args:
            num_envs (int): The number of environments.
            device (str): The device on which the tensors are stored.
        """

        self._num_envs = num_envs
        self._device = device

        # Dimensions of the observation tensors
        self._dim_orientation = (
            2  # theta heading in the world frame (cos(theta), sin(theta)) [0:2]
        )
        self._dim_velocity = 2  # velocity in the world (x_dot, y_dot) [2:4]
        self._dim_omega = 1  # rotation velocity (theta_dot) [4]
        self._dim_task_label = 1  # label of the task to be executed (int) [5]
        self._dim_task_data = 20  # data to be used to fullfil the task (floats) [6:16]

        # Observation buffers
        self._num_observations = 26
        self._obs_buffer = torch.zeros(
            (self._num_envs, self._num_observations),
            device=self._device,
            dtype=torch.float32,
        )
        self._task_label = torch.ones(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_data = torch.zeros(
            (self._num_envs, self._dim_task_data),
            device=self._device,
            dtype=torch.float32,
        )

    def update_observation_tensor(self, current_state: dict) -> torch.Tensor:
        """
        Updates the observation tensor with the current state of the robot.

        Args:
            current_state (dict): The current state of the robot.

        Returns:
            torch.Tensor: The observation tensor.
        """

        self._obs_buffer[:, 0:2] = current_state["orientation"]
        self._obs_buffer[:, 2:4] = current_state["linear_velocity"]
        self._obs_buffer[:, 4] = current_state["angular_velocity"]
        self._obs_buffer[:, 5] = self._task_label
        self._obs_buffer[:, 6:26] = self._task_data
        return self._obs_buffer

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task.

        Args:
            stats (dict): The dictionary to store the statistics.

        Returns:
            dict: The dictionary containing the statistics.
        """

        raise NotImplementedError

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.

        Args:
            current_state (dict): The current state of the robot.

        Returns:
            torch.Tensor: The observation tensor.
        """

        raise NotImplementedError

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.

        Returns:
            torch.Tensor: The reward for the current state of the robot.
        """

        raise NotImplementedError

    def update_kills(self) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not.
        """

        raise NotImplementedError

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict):The new stastistics to be logged.

        Returns:
            dict: The statistics of the training
        """

        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
        """

        raise NotImplementedError

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
    ) -> list:
        """
        Generates a random goal for the task.
        Args:
            env_ids (torch.Tensor): The ids of the environments.
            target_positions (torch.Tensor): The target positions.
            target_orientations (torch.Tensor): The target orientations.

        Returns:
            list: The target positions and orientations.
        """

        raise NotImplementedError

    def get_initial_conditions(
        self,
        env_ids: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates the initial conditions for the robots following a curriculum.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The initial position,
            orientation and velocity of the robot.
        """

        raise NotImplementedError

    def generate_target(self, path, position):
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.

        Args:
            path (str): The path where the pin is to be generated.
            position (torch.Tensor): The position of the target.
        """

        raise NotImplementedError

    def add_visual_marker_to_scene(self, scene: Usd.Stage) -> Tuple[Usd.Stage, None]:
        """
        Adds the visual marker to the scene.

        Args:
            scene (Usd.Stage): The scene to add the visual marker to.

        Returns:
            Tuple[Usd.Stage, None]: The scene and the visual marker.
        """

        raise NotImplementedError


class TaskDict:
    """
    A class to store the task dictionary. It is used to pass the task data to the task class.
    """

    def __init__(self) -> None:
        self.gotoxy = 0
        self.gotopose = 1
        self.trackxyvel = 2
        self.trackxyovel = 3
        self.trackxyvelheading = 4
