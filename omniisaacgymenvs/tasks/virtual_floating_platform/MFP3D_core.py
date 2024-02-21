__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

import torch
from dataclasses import dataclass

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_core import Core as Core2D

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


def quat_to_mat(quat: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions to a batch of rotation matrices.

    Args:
        quat (torch.Tensor): batch of quaternions.

    Returns:
        torch.Tensor: the batch of rotation matrices.
    """

    w, x, y, z = torch.unbind(quat, -1)
    two_s = 2.0 / ((quat * quat).sum(-1) + EPS)
    R = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        -1,
    )
    return R.reshape(quat.shape[:-1] + (3, 3))


def axis_angle_rotation(angle: torch.Tensor, axis: str) -> torch.Tensor:
    """
    Returns the rotation matrix for a given angle and axis.

    Args:
        angle (torch.Tensor): the angle of rotation.
        axis (str): the axis of rotation.

    Returns:
        torch.Tensor: the rotation matrix.
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Converts a batch of euler angles to a batch of rotation matrices.

    Args:
        euler_angles (torch.Tensor): batch of euler angles.
        convention (str): the convention to use for the conversion.

    Returns:
        torch.Tensor: the batch of rotation matrices.
    """

    matrices = [
        axis_angle_rotation(e, c)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[2], matrices[1]), matrices[0])


class Core(Core2D):
    """
    The base class that implements the core of the task.
    """

    def __init__(self, num_envs: int, device: str) -> None:
        """
        Initializes the core of the task.

        Args:
            num_envs (int): number of environments.
            device (str): device to run the code on.
        """

        self._num_envs = num_envs
        self._device = device

        # Dimensions of the observation tensors
        self._dim_orientation: (
            6  # theta heading in the world frame (cos(theta), sin(theta)) [0:6]
        )
        self._dim_velocity: 3  # velocity in the world (x_dot, y_dot) [6:9]
        self._dim_omega: 3  # rotation velocity (theta_dot) [9:12]
        self._dim_task_label: 1  # label of the task to be executed (int) [12]
        self._dim_task_data: 9  # data to be used to fullfil the task (floats) [13:22]

        # Observation buffers
        self._num_observations = 22
        self._obs_buffer = torch.zeros(
            (self._num_envs, self._num_observations),
            device=self._device,
            dtype=torch.float32,
        )
        self._task_label = torch.ones(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_data = torch.zeros(
            (self._num_envs, 9), device=self._device, dtype=torch.float32
        )

    def update_observation_tensor(self, current_state: dict) -> torch.Tensor:
        """
        Updates the observation tensor with the current state of the robot.

        Args:
            current_state (dict): the current state of the robot.

        Returns:
            torch.Tensor: the observation tensor.
        """

        self._obs_buffer[:, 0:6] = current_state["orientation"][:, :2, :].reshape(
            self._num_envs, 6
        )
        self._obs_buffer[:, 6:9] = current_state["linear_velocity"]
        self._obs_buffer[:, 9:12] = current_state["angular_velocity"]
        self._obs_buffer[:, 12] = self._task_label
        self._obs_buffer[:, 13:] = self._task_data
        return self._obs_buffer


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
