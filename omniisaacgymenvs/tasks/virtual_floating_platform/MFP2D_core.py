__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import torch
from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class Core:
    """
    The base class that implements the core of the task."""

    def __init__(self, num_envs: int, device: str) -> None:
        self._num_envs = num_envs
        self._device = device

        # Dimensions of the observation tensors
        self._dim_orientation: 2  # theta heading in the world frame (cos(theta), sin(theta)) [0:2]
        self._dim_velocity: 2  # velocity in the world (x_dot, y_dot) [2:4]
        self._dim_omega: 1  # rotation velocity (theta_dot) [4]
        self._dim_task_label: 1  # label of the task to be executed (int) [5]
        self._dim_task_data: 4  # data to be used to fullfil the task (floats) [6:10]

        # Observation buffers
        self._num_observations = 10
        self._obs_buffer = torch.zeros(
            (self._num_envs, self._num_observations),
            device=self._device,
            dtype=torch.float32,
        )
        self._task_label = torch.ones(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_data = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
        )

    def update_observation_tensor(self, current_state: dict) -> torch.Tensor:
        """
        Updates the observation tensor with the current state of the robot."""

        self._obs_buffer[:, 0:2] = current_state["orientation"]
        self._obs_buffer[:, 2:4] = current_state["linear_velocity"]
        self._obs_buffer[:, 4] = current_state["angular_velocity"]
        self._obs_buffer[:, 5] = self._task_label
        self._obs_buffer[:, 6:10] = self._task_data
        return self._obs_buffer

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task."""

        raise NotImplementedError

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.""" ""

        raise NotImplementedError

    def compute_reward(
        self, current_state: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot."""

        raise NotImplementedError

    def update_kills(self) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not."""

        raise NotImplementedError

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics."""

        raise NotImplementedError

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task."""

        raise NotImplementedError

    def get_goals(
        self,
        env_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_orientations: torch.Tensor,
    ) -> list:
        """
        Generates a random goal for the task."""

        raise NotImplementedError

    def get_spawns(
        self,
        env_ids: torch.Tensor,
        initial_position: torch.Tensor,
        initial_orientation: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        Generates spawning positions for the robots following a curriculum."""

        raise NotImplementedError

    def generate_target(self, path, position):
        """
        Generates a visual marker to help visualize the performance of the agent from the UI.
        """

        raise NotImplementedError

    def add_visual_marker_to_scene(self):
        """
        Adds the visual marker to the scene."""

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


def parse_data_dict(
    dataclass: dataclass, data: dict, ask_for_validation: bool = False
) -> dataclass:
    """
    Parses a dictionary and stores the values in a dataclass."""

    unknown_keys = []
    for key in data.keys():
        if key in dataclass.__dict__.keys():
            dataclass.__setattr__(key, data[key])
        else:
            unknown_keys.append(key)
    try:
        dataclass.__post_init__()
    except:
        pass

    print("Parsed configuration parameters:")
    for key in dataclass.__dict__:
        print("     + " + key + ":" + str(dataclass.__getattribute__(key)))
    if unknown_keys:
        print("The following keys were given but do not match any parameters:")
        for i, key in enumerate(unknown_keys):
            print("     + " + str(i) + " : " + key)
    if ask_for_validation:
        lock = input("Press enter to validate.")
    return dataclass
