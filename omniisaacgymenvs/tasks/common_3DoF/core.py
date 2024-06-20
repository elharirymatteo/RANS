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

def quat_addition(q1, q2):
    q3 = torch.zeros_like(q1)
    q3[:, 0] = (
        q1[:, 0] * q2[:, 0]
        - q1[:, 1] * q2[:, 1]
        - q1[:, 2] * q2[:, 2]
        - q1[:, 3] * q2[:, 3]
    )
    q3[:, 1] = (
        q1[:, 0] * q2[:, 1]
        + q1[:, 1] * q2[:, 0]
        + q1[:, 2] * q2[:, 3]
        - q1[:, 3] * q2[:, 2]
    )
    q3[:, 2] = (
        q1[:, 0] * q2[:, 2]
        - q1[:, 1] * q2[:, 3]
        + q1[:, 2] * q2[:, 0]
        + q1[:, 3] * q2[:, 1]
    )
    q3[:, 3] = (
        q1[:, 0] * q2[:, 3]
        + q1[:, 1] * q2[:, 2]
        - q1[:, 2] * q2[:, 1]
        + q1[:, 3] * q2[:, 0]
    )
    q3 /= torch.norm(q3 + EPS, dim=-1, keepdim=True)
    return q3

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

    def define_observation_space(self, dim_task_data: int):
        """
        Define the observation space dimensions.

        Args:
            dim_observation (int): Dimension of the observation.
            dim_task_data (int): Dimension of the task-specific data part of the observation.
        """

        self._num_observations = (
            self._dim_orientation + self._dim_velocity + self._dim_omega + self._dim_task_label + dim_task_data
        )
        self._dim_task_data = dim_task_data
        
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
        self._obs_buffer[:, 6:6 + self._dim_task_data] = self._task_data
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
        raise NotImplementedError

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError

    def update_kills(self) -> torch.Tensor:
        raise NotImplementedError

    def update_statistics(self, stats: dict) -> dict:
        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        raise NotImplementedError

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
    ) -> list:
        raise NotImplementedError

    def get_initial_conditions(
        self,
        env_ids: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def generate_target(self, path, position):
        raise NotImplementedError

    def add_visual_marker_to_scene(self, scene: Usd.Stage) -> Tuple[Usd.Stage, None]:
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