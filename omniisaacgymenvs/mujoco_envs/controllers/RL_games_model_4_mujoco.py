__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from typing import Dict
from gym import spaces
import numpy as np
import torch
import yaml

from rl_games.algos_torch.players import (
    BasicPpoPlayerContinuous,
    BasicPpoPlayerDiscrete,
)


class RLGamesModel:
    """
    This class implements a wrapper for the RLGames model.
    It is used to interface the RLGames model with the MuJoCo environment.
    It currently only supports PPO agents."""

    def __init__(
        self,
        config: Dict = None,
        config_path: str = None,
        model_path: str = None,
        **kwargs
    ):
        """
        Initialize the RLGames model.

        Args:
            config (Dict, optional): A dictionary containing the configuration of the RLGames model. Defaults to None.
            config_path (str, optional): A string containing the path to the configuration file of the RLGames model. Defaults to None.
            model_path (str, optional): A string containing the path to the model of the RLGames model. Defaults to None.
            **kwargs: Additional arguments."""

        self.obs = dict(
            {
                "state": torch.zeros((1, 10), dtype=torch.float32, device="cuda"),
                "transforms": torch.zeros(5, 8, device="cuda"),
                "masks": torch.zeros(8, dtype=torch.float32, device="cuda"),
                "masses": torch.zeros(3, dtype=torch.float32, device="cuda"),
            }
        )

        # Build model using the configuration files
        if config is None:
            self.loadConfig(config_path)
        else:
            self.cfg = config
        self.buildModel()
        self.restore(model_path)

        # Default target and task values
        self.mode = 0
        self.position_target = [0, 0, 0]
        self.orientation_target = [1, 0, 0, 0]
        self.linear_velocity_target = [0, 0, 0]
        self.angular_velocity_target = [0, 0, 0]

        self.obs_state = torch.zeros((1, 10), dtype=torch.float32, device="cuda")

    def buildModel(self) -> None:
        """
        Build the RLGames model."""

        act_space = spaces.Tuple([spaces.Discrete(2)] * 8)
        obs_space = spaces.Dict(
            {
                "state": spaces.Box(np.ones(10) * -np.Inf, np.ones(10) * np.Inf),
                "transforms": spaces.Box(low=-1, high=1, shape=(8, 5)),
                "masks": spaces.Box(low=0, high=1, shape=(8,)),
                "masses": spaces.Box(low=-np.Inf, high=np.Inf, shape=(3,)),
            }
        )
        self.player = BasicPpoPlayerDiscrete(
            self.cfg, obs_space, act_space, clip_actions=False, deterministic=True
        )

    def loadConfig(self, config_name: str) -> None:
        """
        Load the configuration file of the RLGames model.

        Args:
            config_name (str): A string containing the path to the configuration file of the RLGames model.
        """

        with open(config_name, "r") as stream:
            self.cfg = yaml.safe_load(stream)

    def restore(self, model_name: str) -> None:
        """
        Restore the weights of the RLGames model.

        Args:
            model_name (str): A string containing the path to the checkpoint of an RLGames model matching the configuation file.
        """

        self.player.restore(model_name)
    
    def setTarget(
        self,
        target_position=None,
        target_heading=None,
        target_linear_velocity=None,
        target_angular_velocity=None,
        mode=None,
    ) -> None:
        """
        Set the targets of the agent. mode is task flag.

        Args:
            target_position (list, optional): A list containing the target position. Defaults to None.
            target_heading (list, optional): A list containing the target heading. Defaults to None.
            target_linear_velocity (list, optional): A list containing the target linear velocity. Defaults to None.
            target_angular_velocity (list, optional): A list containing the target angular velocity. Defaults to None.
            mode (int, optional): An integer indicating the agent's task. Defaults to None.
        """

        if mode == 0:
            self.position_target = target_position
            self.mode = mode
        elif mode == 1:
            self.position_target = target_position
            self.orientation_target = target_heading
            self.mode = mode
        elif mode == 2:
            self.linear_velocity_target = target_linear_velocity
            self.mode = mode
        elif mode == 3:
            self.linear_velocity_target = target_linear_velocity
            self.angular_velocity_target = target_angular_velocity
            self.mode = mode
        elif mode == 4:
            self.linear_velocity_target = target_linear_velocity
            self.orientation_target = target_heading
            self.mode = mode
        elif mode == 6:
            #TODO: remove hardcoding
            fp_footprint_radius = 0.31+0.01
            siny_cosp = 2 * target_heading[0] * target_heading[3]
            cosy_cosp = 1 - 2 * (target_heading[3] * target_heading[3])
            target_heading_angle = np.arctan2(siny_cosp, cosy_cosp)
            
            target_position_clone = target_position.copy()
            target_position_clone[0] += fp_footprint_radius * np.cos(target_heading_angle)
            target_position_clone[1] += fp_footprint_radius * np.sin(target_heading_angle)
            
            self.position_target = target_position_clone
            self.orientation_target = target_heading
            self.mode = mode
        else:
            raise ValueError("Please specify a task flag.")

    def generate_task_data(self, state: Dict[str, np.ndarray]) -> None:
        """
        Generate the task data used by the agent.
        The task flag is used to determine the format of the task data.

        Args:
            state (Dict[str, np.ndarray]): A dictionary containing the state of the environment.
        """

        if self.mode == 0:
            self.target = [
                self.position_target[0] - state["position"][0],
                self.position_target[1] - state["position"][1],
                0,
                0,
            ]
        elif self.mode == 1:
            siny_cosp_target = 2 * (
                self.orientation_target[0] * self.orientation_target[3]
                + self.orientation_target[1] * self.orientation_target[2]
            )
            cosy_cosp_target = 1 - 2 * (
                self.orientation_target[2] * self.orientation_target[2]
                + self.orientation_target[3] * self.orientation_target[3]
            )
            heading_target = np.arctan2(siny_cosp_target, cosy_cosp_target)
            siny_cosp_system = 2 * (
                state["quaternion"][0] * state["quaternion"][3]
                + state["quaternion"][1] * state["quaternion"][2]
            )
            cosy_cosp_system = 1 - 2 * (
                state["quaternion"][2] * state["quaternion"][2]
                + state["quaternion"][3] * state["quaternion"][3]
            )
            heading_system = np.arctan2(siny_cosp_system, cosy_cosp_system)
            heading_error = np.arctan2(
                np.sin(heading_target - heading_system),
                np.cos(heading_target - heading_system),
            )
            self.target = [
                self.position_target[0] - state["position"][0],
                self.position_target[1] - state["position"][1],
                np.cos(heading_error),
                np.sin(heading_error),
            ]
        elif self.mode == 2:
            self.target = [
                self.linear_velocity_target[0] - state["linear_velocity"][0],
                self.linear_velocity_target[1] - state["linear_velocity"][1],
                0,
                0,
            ]
        elif self.mode == 3:
            self.target = [
                self.linear_velocity_target[0] - state["linear_velocity"][0],
                self.linear_velocity_target[1] - state["linear_velocity"][1],
                self.angular_velocity_target[2] - state["angular_velocity"][2],
                0,
            ]
        elif self.mode == 4:
            siny_cosp_target = 2 * (
                self.orientation_target[0] * self.orientation_target[3]
                + self.orientation_target[1] * self.orientation_target[2]
            )
            cosy_cosp_target = 1 - 2 * (
                self.orientation_target[2] * self.orientation_target[2]
                + self.orientation_target[3] * self.orientation_target[3]
            )
            heading_target = np.arctan2(siny_cosp_target, cosy_cosp_target)
            siny_cosp_system = 2 * (
                state["quaternion"][0] * state["quaternion"][3]
                + state["quaternion"][1] * state["quaternion"][2]
            )
            cosy_cosp_system = 1 - 2 * (
                state["quaternion"][2] * state["quaternion"][2]
                + state["quaternion"][3] * state["quaternion"][3]
            )
            heading_system = np.arctan2(siny_cosp_system, cosy_cosp_system)
            heading_error = np.arctan2(
                np.sin(heading_target - heading_system),
                np.cos(heading_target - heading_system),
            )
            self.target = [
                self.linear_velocity_target[0] - state["linear_velocity"][0],
                self.linear_velocity_target[1] - state["linear_velocity"][1],
                np.cos(heading_error),
                np.sin(heading_error),
            ]
        
        elif self.mode == 6:
            #TODO: remove hardcoding
            target_to_cone_dist = -2.0
            siny_cosp_target = 2 * (
                self.orientation_target[0] * self.orientation_target[3]
                + self.orientation_target[1] * self.orientation_target[2]
            )
            cosy_cosp_target = 1 - 2 * (
                self.orientation_target[2] * self.orientation_target[2]
                + self.orientation_target[3] * self.orientation_target[3]
            )
            heading_target = np.arctan2(siny_cosp_target, cosy_cosp_target)
            
            anchor_positions = self.position_target.copy()
            anchor_positions[0] += target_to_cone_dist * np.cos(heading_target)
            anchor_positions[1] += target_to_cone_dist * np.sin(heading_target)
            goal_headings = np.arctan2(
                anchor_positions[1] - state["position"][1],
                anchor_positions[0] - state["position"][0]
            )
            siny_cosp_system = 2 * (
                state["quaternion"][0] * state["quaternion"][3]
                + state["quaternion"][1] * state["quaternion"][2]
            )
            cosy_cosp_system = 1 - 2 * (
                state["quaternion"][2] * state["quaternion"][2]
                + state["quaternion"][3] * state["quaternion"][3]
            )
            heading_system = np.arctan2(siny_cosp_system, cosy_cosp_system)
            heading_error = np.abs(
                np.arctan2(
                    np.sin(goal_headings - heading_system),
                    np.cos(goal_headings - heading_system),
                )
            )
            
            self.target = [
                self.position_target[0] - state["position"][0],
                self.position_target[1] - state["position"][1],
                np.cos(heading_error),
                np.sin(heading_error),
            ]

    def makeObservationBuffer(self, state: Dict[str, np.ndarray]) -> None:
        """
        Make the observation buffer used by the agent.

        Args:
            state (Dict[str, np.ndarray]): A dictionary containing the state of the environment.
        """

        self.generate_task_data(state)
        siny_cosp = 2 * (
            state["quaternion"][0] * state["quaternion"][3]
            + state["quaternion"][1] * state["quaternion"][2]
        )
        cosy_cosp = 1 - 2 * (
            state["quaternion"][2] * state["quaternion"][2]
            + state["quaternion"][3] * state["quaternion"][3]
        )
        self.obs_state[0, :2] = torch.tensor(
            [cosy_cosp, siny_cosp], dtype=torch.float32, device="cuda"
        )
        self.obs_state[0, 2:4] = torch.tensor(
            state["linear_velocity"][:2], dtype=torch.float32, device="cuda"
        )
        self.obs_state[0, 4] = state["angular_velocity"][2]
        self.obs_state[0, 5] = self.mode
        self.obs_state[0, 6:] = torch.tensor(
            self.target, dtype=torch.float32, device="cuda"
        )

    def getAction(self, state, is_deterministic=True, **kwargs) -> np.ndarray:
        """
        Get the action of the agent.

        Args:
            state (Dict[str, np.ndarray]): A dictionary containing the state of the environment.
            is_deterministic (bool): A boolean indicating whether the action should be deterministic or not.
            **kwargs: Additional arguments.

        Returns:
            np.ndarray: The action of the agent."""

        self.makeObservationBuffer(state)
        self.obs["state"] = self.obs_state
        actions = (
            self.player.get_action(self.obs.copy(), is_deterministic=is_deterministic)
            .cpu()
            .numpy()
        )
        return actions