__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


from typing import List, Tuple, Dict, Union
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import os

from omniisaacgymenvs.mujoco_envs.controllers.discrete_LQR_controller import (
    DiscreteController,
)
from omniisaacgymenvs.mujoco_envs.controllers.RL_games_model_4_mujoco import (
    RLGamesModel,
)


class BaseController:
    """
    Base class for high-level controllers."""

    def __init__(self, dt: float, save_dir: str = "mujoco_experiment") -> None:
        """
        Initializes the controller.

        Args:
            dt (float): Simulation time step.
            save_dir (str, optional): Directory to save the simulation data. Defaults to "mujoco_experiment".
        """

        self.save_dir = save_dir
        self.dt = dt
        self.time = 0

        self.csv_datas = []
        self.initializeLoggers()

    def initializeLoggers(self) -> None:
        """
        Initializes the loggers for the simulation.
        Allowing for the simulation to be replayed/plotted."""

        self.logs = {}
        self.logs["timevals"] = []
        self.logs["angular_velocity"] = []
        self.logs["linear_velocity"] = []
        self.logs["position"] = []
        self.logs["quaternion"] = []
        self.logs["actions"] = []

    def updateLoggers(self, state: Dict[str, np.ndarray], action: np.ndarray) -> None:
        """
        Updates the loggers for the simulation.

        Args:
            state (Dict[str, np.ndarray]): State of the system.
            action (np.ndarray): Action taken by the controller."""

        self.logs["timevals"].append(self.time)
        self.logs["position"].append(state["position"])
        self.logs["quaternion"].append(state["quaternion"])
        self.logs["angular_velocity"].append(state["angular_velocity"])
        self.logs["linear_velocity"].append(state["linear_velocity"])
        self.logs["actions"].append(action)
        self.time += self.dt

    def isDone(self) -> bool:
        """
        Checks if the simulation is done.

        Returns:
            bool: True if the simulation is done, False otherwise."""

        return False

    def getGoal(self) -> None:
        """
        Returns the current goal of the controller."""

        raise NotImplementedError

    def setGoal(self) -> None:
        """
        Sets the goal of the controller."""

        raise NotImplementedError

    def getAction(self, **kwargs) -> np.ndarray:
        """
        Gets the action from the controller."""

        raise NotImplementedError

    def plotSimulation(
        self, dpi: int = 120, width: int = 600, height: int = 800
    ) -> None:
        """
        Plots the simulation.

        Args:
            dpi (int, optional): Dots per inch. Defaults to 120.
            width (int, optional): Width of the figure. Defaults to 600.
            height (int, optional): Height of the figure. Defaults to 800."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

        ax[0].plot(self.logs["timevals"], self.logs["angular_velocity"])
        ax[0].set_title("angular velocity")
        ax[0].set_ylabel("radians / second")

        ax[1].plot(self.logs["timevals"], self.logs["linear_velocity"])
        ax[1].set_xlabel("time (seconds)")
        ax[1].set_ylabel("meters / second")
        _ = ax[1].set_title("linear_velocity")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "velocities.png"))
        except Exception as e:
            print("Saving failed: ", e)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        ax[0].plot(self.logs["timevals"], np.abs(self.logs["position"]))
        ax[0].set_xlabel("time (seconds)")
        ax[0].set_ylabel("meters")
        _ = ax[0].set_title("position")
        ax[0].set_yscale("log")

        ax[1].plot(
            np.array(self.logs["position"])[:, 0], np.array(self.logs["position"])[:, 1]
        )
        ax[1].set_xlabel("meters")
        ax[1].set_ylabel("meters")
        _ = ax[1].set_title("x y coordinates")
        plt.tight_layout()
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "positions.png"))
        except Exception as e:
            print("Saving failed: ", e)

    def saveSimulationData(self, suffix: str = "") -> None:
        """
        Saves the simulation data.

        Args:
            suffix (str, optional): Suffix to add to the file name. Defaults to ""."""

        var_name = ["x", "y", "z", "w"]
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            csv_data = pd.DataFrame()
            for key in self.logs.keys():
                if len(self.logs[key]) != 0:
                    if key == "actions":
                        data = np.array(self.logs[key])
                        for i in range(data.shape[1]):
                            csv_data["t_" + str(i)] = data[:, i]
                    else:
                        data = np.array(self.logs[key])
                        if len(data.shape) > 1:
                            for i in range(data.shape[1]):
                                csv_data[var_name[i] + "_" + key] = data[:, i]
                        else:
                            csv_data[key] = data
            csv_data.to_csv(os.path.join(self.save_dir, "exp_logs" + suffix + ".csv"))
            self.csv_datas.append(csv_data)

        except Exception as e:
            print("Saving failed: ", e)

    def plotBatch(self, dpi: int = 120, width: int = 600, height: int = 800) -> None:
        """
        Plots a batch of simulations.

        Args:
            dpi (int, optional): Dots per inch. Defaults to 120.
            width (int, optional): Width of the figure. Defaults to 600.
            height (int, optional): Height of the figure. Defaults to 800."""

        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize=figsize)
        for csv_data in self.csv_datas:
            plt.plot(csv_data["x_position"], csv_data["y_position"])
        plt.axis("equal")
        plt.xlabel("meters")
        plt.ylabel("meters")
        plt.tight_layout()
        fig.savefig(os.path.join(self.save_dir, "positions.png"))


class PoseController(BaseController):
    """
    Controller for the pose of the robot."""

    def __init__(
        self,
        dt: float,
        model: Union[RLGamesModel, DiscreteController],
        goals_x: List[float],
        goals_y: List[float],
        goals_theta: List[float],
        position_distance_threshold: float = 0.03,
        orientation_distance_threshold: float = 0.03,
        save_dir: str = "mujoco_experiment",
        **kwargs
    ) -> None:
        """
        Initializes the controller.

        Args:
            dt (float): Simulation time step.
            model (Union[RLGamesModel, DiscreteController]): Low-level controller.
            goals_x (List[float]): List of x coordinates of the goals.
            goals_y (List[float]): List of y coordinates of the goals.
            goals_theta (List[float]): List of theta coordinates of the goals.
            position_distance_threshold (float, optional): Distance threshold for the position. Defaults to 0.03.
            orientation_distance_threshold (float, optional): Distance threshold for the orientation. Defaults to 0.03.
            save_dir (str, optional): Directory to save the simulation data. Defaults to "mujoco_experiment".
            **kwargs: Additional arguments."""

        super().__init__(dt, save_dir)
        # Discrete controller
        self.model = model
        # Creates an array goals
        if goals_theta is None:
            goals_theta = np.zeros_like(goals_x)
        self.goals = np.array([goals_x, goals_y, goals_theta]).T

        self.current_goal = self.goals[0]
        self.current_goal_controller = np.zeros((3), dtype=np.float32)
        self.current_goal_controller = self.current_goal

        self.position_distance_threshold = position_distance_threshold
        self.orientation_distance_threshold = orientation_distance_threshold

    def initializeLoggers(self) -> None:
        """
        Initializes the loggers."""

        super().initializeLoggers()
        self.logs["position_target"] = []
        self.logs["heading_target"] = []

    def updateLoggers(self, state, actions) -> None:
        """
        Updates the loggers.

        Args:
            state (Dict[str, np.ndarray]): State of the system.
            actions (np.ndarray): Action taken by the controller."""

        super().updateLoggers(state, actions)
        self.logs["position_target"].append(self.current_goal[:2])
        self.logs["heading_target"].append(self.current_goal[-1])

    def isGoalReached(self, state: Dict[str, np.ndarray]) -> bool:
        """
        Checks if the goal is reached.

        Args:
            state (Dict[str, np.ndarray]): State of the system.

        Returns:
            bool: True if the goal is reached, False otherwise."""

        dist = np.linalg.norm(self.current_goal[:2] - state["position"][:2])
        if dist < self.position_distance_threshold:
            return True

    def getGoal(self) -> np.ndarray:
        """
        Returns the current goal.

        Returns:
            np.ndarray: Current goal."""

        return self.current_goal

    def setGoal(self, goal: np.ndarray) -> None:
        """
        Sets the goal of the controller.

        Args:
            goal (np.ndarray): Goal to set."""

        self.current_goal = goal
        self.goals = np.array([goal])

    def isDone(self) -> bool:
        """
        Checks if the simulation is done.

        Returns:
            bool: True if the simulation is done, False otherwise."""

        return len(self.goals) == 0

    def setTarget(self) -> None:
        """
        Sets the target of the low-level controller."""

        position_goal = self.current_goal
        yaw = self.current_goal[2]
        q = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        orientation_goal = q
        self.model.setTarget(
            target_position=position_goal, target_heading=orientation_goal
        )

    def getAction(
        self, state: Dict[str, np.ndarray], is_deterministic: bool = True
    ) -> np.ndarray:
        """
        Gets the action from the controller.

        Args:
            state (Dict[str, np.ndarray]): State of the system.
            is_deterministic (bool, optional): Whether the action is deterministic or not. Defaults to True.

        Returns:
            np.ndarray: Action taken by the controller."""

        if self.isGoalReached(state):
            print("Goal reached!")
            if len(self.goals) > 1:
                self.current_goal = self.goals[1]
                self.goals = self.goals[1:]
            else:
                self.goals = []
        self.setTarget()
        actions = self.model.getAction(state, is_deterministic=is_deterministic)
        self.updateLoggers(state, actions)
        return actions

    def plotSimulation(
        self, dpi: int = 90, width: int = 1000, height: int = 1000
    ) -> None:
        """
        Plots the simulation.

        Args:
            dpi (int, optional): Dots per inch. Defaults to 90.
            width (int, optional): Width of the figure. Defaults to 1000.
            height (int, optional): Height of the figure. Defaults to 1000."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

        ax[0].plot(self.logs["timevals"], self.logs["angular_velocity"])
        ax[0].set_title("angular velocity")
        ax[0].set_ylabel("radians / second")

        ax[1].plot(
            self.logs["timevals"],
            self.logs["linear_velocity"],
            label="system velocities",
        )
        ax[1].legend()
        ax[1].set_xlabel("time (seconds)")
        ax[1].set_ylabel("meters / second")
        _ = ax[1].set_title("linear_velocity")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "velocities.png"))
        except Exception as e:
            print("Saving failed: ", e)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.scatter(
            np.array(self.logs["position_target"])[:, 0],
            np.array(self.logs["position_target"])[:, 1],
            label="position goals",
        )
        ax.plot(
            np.array(self.logs["position"])[:, 0],
            np.array(self.logs["position"])[:, 1],
            label="system position",
        )
        ax.legend()
        ax.set_xlabel("meters")
        ax.set_ylabel("meters")
        ax.axis("equal")
        _ = ax.set_title("x y coordinates")
        plt.tight_layout()
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "positions.png"))
        except Exception as e:
            print("Saving failed: ", e)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.plot(
            self.logs["timevals"], np.array(self.logs["actions"]), label="system action"
        )
        plt.tight_layout()
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "actions.png"))
        except Exception as e:
            print("Saving failed: ", e)


class PositionController(BaseController):
    def __init__(
        self,
        dt: float,
        model: Union[RLGamesModel, DiscreteController],
        goals_x: List[float],
        goals_y: List[float],
        position_distance_threshold: float = 0.03,
        save_dir: str = "mujoco_experiment",
        **kwargs
    ) -> None:
        """
        Initializes the controller.

        Args:
            dt (float): Simulation time step.
            model (Union[RLGamesModel, DiscreteController]): Low-level controller.
            goals_x (List[float]): List of x coordinates of the goals.
            goals_y (List[float]): List of y coordinates of the goals.
            position_distance_threshold (float, optional): Distance threshold for the position. Defaults to 0.03.
            save_dir (str, optional): Directory to save the simulation data. Defaults to "mujoco_experiment".
            **kwargs: Additional arguments."""

        super().__init__(dt, save_dir)
        self.model = model
        self.goals = np.array([goals_x, goals_y, [0] * len(goals_x)]).T
        self.current_goal = self.goals[0]
        self.distance_threshold = position_distance_threshold

    def initializeLoggers(self) -> None:
        """
        Initializes the loggers."""

        super().initializeLoggers()
        self.logs["position_target"] = []

    def updateLoggers(self, state, actions) -> None:
        """
        Updates the loggers.

        Args:
            state (Dict[str, np.ndarray]): State of the system.
            actions (np.ndarray): Action taken by the controller."""

        super().updateLoggers(state, actions)
        self.logs["position_target"].append(self.current_goal[:2])

    def isGoalReached(self, state: Dict[str, np.ndarray]) -> bool:
        """
        Checks if the goal is reached.

        Args:
            state (Dict[str, np.ndarray]): State of the system.

        Returns:
            bool: True if the goal is reached, False otherwise."""

        dist = np.linalg.norm(self.current_goal[:2] - state["position"][:2])
        if dist < self.distance_threshold:
            return True

    def getGoal(self) -> np.ndarray:
        """
        Returns the current goal."""

        return self.current_goal

    def setGoal(self, goal) -> None:
        """
        Sets the goal of the controller.

        Args:
            goal (np.ndarray): Goal to set."""

        self.current_goal = goal
        self.goals = np.array([goal])

    def isDone(self) -> bool:
        """
        Checks if the simulation is done.

        Returns:
            bool: True if the simulation is done, False otherwise."""

        return len(self.goals) == 0

    def setTarget(self) -> None:
        """
        Sets the target of the low-level controller."""

        self.model.setTarget(target_position=self.current_goal)

    def getAction(self, state, is_deterministic: bool = True) -> np.ndarray:
        """
        Gets the action from the controller.

        Args:
            state (Dict[str, np.ndarray]): State of the system.
            is_deterministic (bool, optional): Whether the action is deterministic or not. Defaults to True.

        Returns:
            np.ndarray: Action taken by the controller."""

        if self.isGoalReached(state):
            print("Goal reached!")
            if len(self.goals) > 1:
                self.current_goal = self.goals[1]
                self.goals = self.goals[1:]
            else:
                self.goals = []

        self.setTarget()
        actions = self.model.getAction(state, is_deterministic=is_deterministic)
        self.updateLoggers(state, actions)
        return actions

    def plotSimulation(
        self, dpi: int = 90, width: int = 1000, height: int = 1000
    ) -> None:
        """
        Plots the simulation.

        Args:
            dpi (int, optional): Dots per inch. Defaults to 90.
            width (int, optional): Width of the figure. Defaults to 1000.
            height (int, optional): Height of the figure. Defaults to 1000."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

        ax[0].plot(self.logs["timevals"], self.logs["angular_velocity"])
        ax[0].set_title("angular velocity")
        ax[0].set_ylabel("radians / second")

        ax[1].plot(
            self.logs["timevals"],
            self.logs["linear_velocity"],
            label="system velocities",
        )
        ax[1].legend()
        ax[1].set_xlabel("time (seconds)")
        ax[1].set_ylabel("meters / second")
        _ = ax[1].set_title("linear_velocity")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "velocities.png"))
        except Exception as e:
            print("Saving failed: ", e)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.scatter(
            np.array(self.logs["position_target"])[:, 0],
            np.array(self.logs["position_target"])[:, 1],
            label="position goals",
        )
        ax.plot(
            np.array(self.logs["position"])[:, 0],
            np.array(self.logs["position"])[:, 1],
            label="system position",
        )
        ax.legend()
        ax.set_xlabel("meters")
        ax.set_ylabel("meters")
        ax.axis("equal")
        _ = ax.set_title("x y coordinates")
        plt.tight_layout()
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "positions.png"))
        except Exception as e:
            print("Saving failed: ", e)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.plot(
            self.logs["timevals"], np.array(self.logs["actions"]), label="system action"
        )
        plt.tight_layout()
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "actions.png"))
        except Exception as e:
            print("Saving failed: ", e)


class TrajectoryTracker:
    """
    A class to generate and track trajectories."""

    def __init__(
        self, lookahead: float = 0.25, closed: bool = False, offset=(0, 0), **kwargs
    ):
        """
        Initializes the trajectory tracker.

        Args:
            lookahead (float, optional): Lookahead distance. Defaults to 0.25.
            closed (bool, optional): Whether the trajectory is closed or not. Defaults to False.
            offset (tuple, optional): Offset of the trajectory. Defaults to (0,0).
            **kwargs: Additional arguments."""

        self.current_point = -1
        self.lookhead = lookahead
        self.closed = closed
        self.is_done = False
        self.offset = np.array(offset)

    def generateCircle(self, radius: float = 2, num_points: int = 360 * 10):
        """
        Generates a circle trajectory.

        Args:
            radius (float, optional): Radius of the circle. Defaults to 2.
            num_points (int, optional): Number of points. Defaults to 360*10."""

        theta = np.linspace(0, 2 * np.pi, num_points, endpoint=(not self.closed))
        self.positions = (
            np.array([np.cos(theta) * radius, np.sin(theta) * radius]).T + self.offset
        )
        self.angles = np.array([-np.sin(theta), np.cos(theta)]).T

    def generateSquare(self, h: float = 2, num_points: int = 360 * 10) -> None:
        """
        Generates a square trajectory.

        Args:
            h (float, optional): Height of the square. Defaults to 2.
            num_points (int, optional): Number of points. Defaults to 360*10."""

        points_per_side = num_points // 4
        s1y = np.linspace(-h / 2, h / 2, num_points, endpoint=False)
        s1x = np.ones_like(s1y) * h / 2
        s2x = np.linspace(h / 2, -h / 2, num_points, endpoint=False)
        s2y = np.ones_like(s2x) * h / 2
        s3y = np.linspace(h / 2, -h / 2, num_points, endpoint=False)
        s3x = np.ones_like(s3y) * (-h / 2)
        s4x = np.linspace(-h / 2, h / 2, num_points, endpoint=False)
        s4y = np.ones_like(s4x) * (-h / 2)
        self.positions = (
            np.vstack(
                [np.hstack([s1x, s2x, s3x, s4x]), np.hstack([s1y, s2y, s3y, s4y])]
            ).T
            + self.offset
        )
        self.angles = np.ones_like(self.positions)

    def generateSpiral(
        self,
        start_radius: float = 0.5,
        end_radius: float = 2,
        num_loop: float = 5,
        num_points: int = 360 * 20,
    ) -> None:
        """
        Generates a spiral trajectory.

        Args:
            start_radius (float, optional): Start radius of the spiral. Defaults to 0.5.
            end_radius (float, optional): End radius of the spiral. Defaults to 2.
            num_loop (float, optional): Number of loops. Defaults to 5.
            num_points (int, optional): Number of points. Defaults to 360*20."""

        radius = np.linspace(
            start_radius, end_radius, num_points, endpoint=(not self.closed)
        )
        theta = np.linspace(
            0, 2 * np.pi * num_loop, num_points, endpoint=(not self.closed)
        )
        self.positions = (
            np.array([np.cos(theta) * radius, np.sin(theta) * radius]).T + self.offset
        )
        self.angles = np.array([-np.sin(theta), np.cos(theta)]).T

    def getTrackingPointIdx(self, position: np.ndarray) -> None:
        """
        Gets the tracking point index.
        The tracking point is the point the robot is currently locked on.

        Args:
            position (np.ndarray): Current position of the robot."""

        distances = np.linalg.norm(self.positions - position, axis=1)
        if self.current_point == -1:
            self.current_point = 0
        else:
            indices = np.where(distances < self.lookhead)[0]
            if len(indices) > 0:
                indices = indices[indices < 60]
                if len(indices) > 0:
                    self.current_point = np.max(indices)

    def rollTrajectory(self) -> None:
        """
        Rolls the trajectory, so that the current point is the first point."""

        if self.closed:
            self.positions = np.roll(self.positions, -self.current_point, axis=0)
            self.angles = np.roll(self.angles, -self.current_point, axis=0)
            self.current_point = 0
        else:
            self.positions = self.positions[self.current_point :]
            self.angles = self.angles[self.current_point :]
            self.current_point = 0
        if self.positions.shape[0] <= 1:
            self.is_done = True

    def isDone(self):
        """
        Checks if the trajectory is done."""

        return self.is_done

    def getPointForTracking(self) -> List[np.ndarray]:
        """
        Gets the position the tracker is currently locked on.

        Returns:
            List[np.ndarray]: Position being tracked."""

        position = self.positions[self.current_point]
        angle = self.angles[self.current_point]
        self.rollTrajectory()
        return position, angle

    def get_target_position(self) -> np.ndarray:
        """
        Gets the target position.

        Returns:
            np.ndarray: Target position."""

        return self.target_position

    def computeVelocityVector(
        self, target_position: np.ndarray, position: np.ndarray
    ) -> np.ndarray:
        """
        Computes the velocity vector.
        That is the vector that will enable the robot to reach the position being tracked.

        Args:
            target_position (np.ndarray): Position being tracked.
            position (np.ndarray): Current position of the robot."""

        diff = target_position - position
        return diff / np.linalg.norm(diff)

    def getVelocityVector(self, position: np.ndarray) -> np.ndarray:
        """
        Gets the velocity vector.

        Args:
            position (np.ndarray): Current position of the robot.

        Returns:
            np.ndarray: Velocity vector."""

        self.getTrackingPointIdx(position)
        self.target_position, target_angle = self.getPointForTracking()
        velocity_vector = self.computeVelocityVector(self.target_position, position)
        return velocity_vector


class VelocityTracker(BaseController):
    def __init__(
        self,
        dt: float,
        model: Union[RLGamesModel, DiscreteController],
        target_tracking_velocity: float = 0.25,
        lookahead_dist: float = 0.15,
        closed: bool = True,
        x_offset: float = 0,
        y_offset: float = 0,
        radius: float = 1.5,
        height: float = 1.5,
        start_radius: float = 0.5,
        end_radius: float = 2.0,
        num_loops: int = 4,
        trajectory_type: str = "circle",
        save_dir: str = "mujoco_experiment",
        **kwargs
    ) -> None:
        """
        Initializes the controller.

        Args:
            dt (float): Simulation time step.
            model (Union[RLGamesModel, DiscreteController]): Low-level controller.
            target_tracking_velocity (float, optional): Target tracking velocity. Defaults to 0.25.
            lookahead_dist (float, optional): Lookahead distance. Defaults to 0.15.
            closed (bool, optional): Whether the trajectory is closed or not. Defaults to True.
            x_offset (float, optional): x offset of the trajectory. Defaults to 0.
            y_offset (float, optional): y offset of the trajectory. Defaults to 0.
            radius (float, optional): Radius of the trajectory. Defaults to 1.5.
            height (float, optional): Height of the trajectory. Defaults to 1.5.
            start_radius (float, optional): Start radius of the trajectory. Defaults to 0.5.
            end_radius (float, optional): End radius of the trajectory. Defaults to 2.0.
            num_loops (int, optional): Number of loops. Defaults to 4.
            trajectory_type (str, optional): Type of trajectory. Defaults to "circle".
            save_dir (str, optional): Directory to save the simulation data. Defaults to "mujoco_experiment".
            **kwargs: Additional arguments."""

        super().__init__(dt, save_dir)
        self.tracker = TrajectoryTracker(
            lookahead=lookahead_dist, closed=closed, offset=(x_offset, y_offset)
        )
        if trajectory_type.lower() == "square":
            self.tracker.generateSquare(h=height)
        elif trajectory_type.lower() == "circle":
            self.tracker.generateCircle(radius=radius)
        elif trajectory_type.lower() == "spiral":
            self.tracker.generateSpiral(
                start_radius=start_radius, end_radius=end_radius, num_loop=num_loops
            )
        else:
            raise ValueError(
                "Unknown trajectory type. Must be square, circle or spiral."
            )

        self.model = model
        self.target_tracking_velocity = target_tracking_velocity
        self.velocity_goal = [0, 0, 0]

    def initializeLoggers(self) -> None:
        """
        Initializes the loggers."""

        super().initializeLoggers()
        self.logs["velocity_goal"] = []
        self.logs["position_target"] = []

    def updateLoggers(self, state, actions) -> None:
        """
        Updates the loggers.

        Args:
            state (Dict[str, np.ndarray]): State of the system.
            actions (np.ndarray): Action taken by the controller."""

        super().updateLoggers(state, actions)
        self.logs["velocity_goal"].append(self.velocity_goal[:2])
        self.logs["position_target"].append(self.getTargetPosition())

    def getGoal(self) -> np.ndarray:
        """
        Returns the current goal.

        Returns:
            np.ndarray: Current goal."""

        return self.velocity_goal

    def setGoal(self, goal: np.ndarray) -> None:
        """
        Sets the goal of the controller.

        Args:
            goal (np.ndarray): Goal to set."""

        self.target_tracking_velocity = goal

    def getTargetPosition(self) -> np.ndarray:
        """
        Gets the target position.

        Returns:
            np.ndarray: Target position."""

        return self.tracker.get_target_position()

    def isDone(self) -> bool:
        """
        Checks if the simulation is done.

        Returns:
            bool: True if the simulation is done, False otherwise."""

        return self.tracker.is_done

    def setTarget(self) -> None:
        """
        Sets the target of the low-level controller."""

        self.model.setTarget(target_linear_velocity=self.velocity_goal)

    def getAction(
        self, state: Dict[str, np.ndarray], is_deterministic: bool = True
    ) -> np.ndarray:
        """
        Gets the action from the controller.

        Args:
            state (Dict[str, np.ndarray]): State of the system.
            is_deterministic (bool, optional): Whether the action is deterministic or not. Defaults to True.
        """

        self.velocity_vector = self.tracker.getVelocityVector(state["position"][:2])
        self.velocity_goal[0] = self.velocity_vector[0] * self.target_tracking_velocity
        self.velocity_goal[1] = self.velocity_vector[1] * self.target_tracking_velocity
        self.setTarget()
        actions = self.model.getAction(state, is_deterministic=is_deterministic)
        self.updateLoggers(state, actions)
        return actions

    def plotSimulation(
        self, dpi: int = 135, width: int = 1000, height: int = 1000
    ) -> None:
        """
        Plots the simulation.

        Args:
            dpi (int, optional): Dots per inch. Defaults to 135.
            width (int, optional): Width of the figure. Defaults to 1000.
            height (int, optional): Height of the figure. Defaults to 1000."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

        ax.plot(
            self.logs["timevals"],
            self.logs["linear_velocity"],
            label="system velocities",
        )
        ax.plot(
            self.logs["timevals"], self.logs["velocity_goal"], label="target velocities"
        )
        ax.legend()
        ax.set_xlabel("time (seconds)")
        ax.set_ylabel("Linear velocities (m/s)")
        _ = ax.set_title("Linear velocity tracking")
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "velocities.png"))
        except Exception as e:
            print("Saving failed: ", e)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.plot(
            np.array(self.logs["position_target"])[:, 0],
            np.array(self.logs["position_target"])[:, 1],
            label="trajectory",
        )
        ax.plot(
            np.array(self.logs["position"])[:, 0],
            np.array(self.logs["position"])[:, 1],
            label="system position",
        )
        ax.legend()
        ax.set_xlabel("x (meters)")
        ax.set_ylabel("y (meters)")
        ax.axis("equal")
        _ = ax.set_title("Trajectory in xy plane")
        plt.tight_layout()
        try:
            os.makedirs(self.save_dir, exist_ok=True)
            fig.savefig(os.path.join(self.save_dir, "positions.png"))
        except Exception as e:
            print("Saving failed: ", e)


class HLControllerFactory:
    """
    Factory for high-level controllers."""

    def __init__(self):
        self.registered_controllers = {}

    def registerController(
        self,
        name: str,
        controller: Union[PositionController, PoseController, VelocityTracker],
    ):
        """
        Registers a controller.

        Args:
            name (str): Name of the controller.
            controller (Union[PositionController, PoseController, VelocityTracker]): Controller class.
        """
        self.registered_controllers[name] = controller

    def parseControllerConfiguration(self, cfg: Dict):
        """
        Parses the controller configuration.

        Args:
            cfg (Dict): Configuration dictionary."""

        return cfg["hl_task"], cfg["hl_task"]["name"]

    def __call__(
        self, cfg: Dict, model: Union[RLGamesModel, DiscreteController], dt: float
    ):
        """
        Creates a controller.

        Args:
            cfg (Dict): Configuration dictionary.
            model (Union[RLGamesModel, DiscreteController]): Low-level controller.
            dt (float): Simulation time step."""

        new_cfg, mode = self.parseControllerConfiguration(cfg)
        assert mode in list(self.registered_controllers.keys()), "Unknown hl_task mode."
        return self.registered_controllers[mode](dt, model, **new_cfg)


"""
Register the controllers."""

hlControllerFactory = HLControllerFactory()
hlControllerFactory.registerController("position", PositionController)
hlControllerFactory.registerController("pose", PoseController)
hlControllerFactory.registerController("linear_velocity", VelocityTracker)
