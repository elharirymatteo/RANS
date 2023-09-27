from typing import List, Tuple, Dict, Union
import numpy as np
import torch

from omniisaacgymenvs.mujoco_envs.controllers.discrete_LQR_controller import DiscreteController
from omniisaacgymenvs.mujoco_envs.controllers.RL_games_model_4_mujoco import RLGamesModel

class PoseController:
    """
    Controller for the pose of the robot."""

    def __init__(self, model: Union[RLGamesModel, DiscreteController],
                       goals_x: List[float],
                       goals_y: List[float],
                       goals_theta: List[float],
                       position_distance_threshold: float = 0.03,
                       orientation_distance_threshold: float = 0.03,
                       **kwargs) -> None:

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

    def isGoalReached(self, state: Dict[str, np.ndarray]) -> bool:
        dist = np.linalg.norm(self.current_goal[:2] - state["position"][:2])
        if dist < self.position_distance_threshold:
            return True
    
    def getGoal(self) -> np.ndarray:
        return self.current_goal
    
    def setGoal(self, goal:np.ndarray) -> None:
        self.current_goal = goal
        self.goals = np.array([goal])

    def isDone(self) -> bool:
        return len(self.goals) == 0
    
    def setTarget(self):
        position_goal = self.current_goal
        yaw = self.current_goal[2]
        q = [np.cos(yaw/2),0,0,np.sin(yaw/2)]
        orientation_goal = q
        self.model.setTarget(target_position=position_goal, target_heading=orientation_goal)
    
    def getAction(self, state: Dict[str, np.ndarray], is_deterministic: bool = True) -> np.ndarray:
        if self.isGoalReached(state):
            print("Goal reached!")
            if len(self.goals) > 1:
                self.current_goal = self.goals[1]
                self.goals = self.goals[1:]   
            else:
                self.goals = []
        self.setTarget()
        return self.model.getAction(state, is_deterministic=is_deterministic)


class PositionController:
    def __init__(self, model: Union[RLGamesModel, DiscreteController],
                       goals_x: List[float],
                       goals_y: List[float],
                       position_distance_threshold: float = 0.03,
                       **kwargs) -> None:
        
        self.model = model
        self.goals = np.array([goals_x, goals_y, [0]*len(goals_x)]).T
        self.current_goal = self.goals[0]
        self.distance_threshold = position_distance_threshold

    def isGoalReached(self, state):
        dist = np.linalg.norm(self.current_goal[:2] - state["position"][:2])
        if dist < self.distance_threshold:
            return True
    
    def getGoal(self):
        return self.current_goal
    
    def setGoal(self, goal):
        self.current_goal = goal
        self.goals = np.array([goal])

    def isDone(self):
        return len(self.goals) == 0

    def setTarget(self):
        self.model.setTarget(target_position=self.current_goal)

    def getAction(self, state, is_deterministic: bool = True):
        if self.isGoalReached(state):
            print("Goal reached!")
            if len(self.goals) > 1:
                self.current_goal = self.goals[1]
                self.goals = self.goals[1:]
            else:
                self.goals = []

        self.setTarget()
        return self.model.getAction(state, is_deterministic=is_deterministic)


class TrajectoryTracker:
    """
    A class to generate and track trajectories."""

    def __init__(self, lookahead:float = 0.25,
                       closed:bool = False,
                       offset = (0,0),
                       **kwargs):

        self.current_point = -1
        self.lookhead = lookahead
        self.closed = closed
        self.is_done = False
        self.offset = np.array(offset)

    def generateCircle(self, radius:float = 2, num_points:int = 360*10):
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=(not self.closed))
        self.positions = np.array([np.cos(theta) * radius, np.sin(theta) * radius]).T + self.offset
        self.angles = np.array([-np.sin(theta), np.cos(theta)]).T

    def generateSquare(self, h:float = 2, num_points:int = 360*10) -> None:
        points_per_side = num_points // 4
        s1y = np.linspace(-h/2,h/2, num_points, endpoint=False)
        s1x = np.ones_like(s1y)*h/2
        s2x = np.linspace(h/2,-h/2, num_points, endpoint=False)
        s2y = np.ones_like(s2x) * h/2
        s3y = np.linspace(h/2,-h/2, num_points, endpoint=False)
        s3x = np.ones_like(s3y) * (-h/2)
        s4x = np.linspace(-h/2,h/2, num_points, endpoint=False)
        s4y = np.ones_like(s4x) * (-h/2)
        self.positions = np.vstack([np.hstack([s1x,s2x,s3x,s4x]), np.hstack([s1y,s2y,s3y,s4y])]).T + self.offset
        self.angles = np.ones_like(self.positions)

    def generateSpiral(self, start_radius:float = 0.5, end_radius:float = 2, num_loop:float = 5, num_points: int = 360*20) -> None:
        radius = np.linspace(start_radius, end_radius, num_points, endpoint=(not self.closed))
        theta = np.linspace(0, 2*np.pi*num_loop, num_points, endpoint=(not self.closed))
        self.positions = np.array([np.cos(theta) * radius, np.sin(theta) * radius]).T + self.offset
        self.angles = np.array([-np.sin(theta), np.cos(theta)]).T
    
    def getTrackingPointIdx(self, position:np.ndarray) -> None:
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
        if self.closed:
            self.positions = np.roll(self.positions, -self.current_point, axis=0)
            self.angles = np.roll(self.angles, -self.current_point, axis=0)
            self.current_point = 0
        else:
            self.positions = self.positions[self.current_point:]
            self.angles = self.angles[self.current_point:]
            self.current_point = 0 
        if self.positions.shape[0] <= 1:
            self.is_done = True

    def isDone(self):
        return self.is_done

    def getPointForTracking(self) -> List[np.ndarray]:
        position = self.positions[self.current_point]
        angle = self.angles[self.current_point]
        self.rollTrajectory()
        return position, angle
    
    def get_target_position(self) -> np.ndarray:
        return self.target_position
    
    def computeVelocityVector(self, target_position:np.ndarray, position:np.ndarray) -> np.ndarray:
        diff = target_position - position
        return diff / np.linalg.norm(diff)
    
    def getVelocityVector(self, position:np.ndarray) -> np.ndarray:
        self.getTrackingPointIdx(position)
        self.target_position, target_angle = self.getPointForTracking()
        velocity_vector = self.computeVelocityVector(self.target_position, position)
        return velocity_vector


class VelocityTracker:
    def __init__(self, model: Union[RLGamesModel, DiscreteController],
                       target_tracking_velocity:float = 0.25,
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
                       **kwargs):
        
        self.tracker = TrajectoryTracker(lookahead=lookahead_dist, closed=closed, offset=(x_offset, y_offset))
        if trajectory_type.lower() == "square":
            self.tracker.generateSquare(h=height)
        elif trajectory_type.lower() == "circle":
            self.tracker.generateCircle(radius=radius)
        elif trajectory_type.lower() == "spiral":
            self.tracker.generateSpiral(start_radius=start_radius, end_radius=end_radius, num_loop=num_loops)
        else:
            raise ValueError("Unknown trajectory type. Must be square, circle or spiral.")

        self.model = model
        self.target_tracking_velocity = target_tracking_velocity
        self.velocity_goal = [0,0,0]
    
    def getGoal(self):
        return self.velocity_goal
    
    def setGoal(self, goal):
        self.target_tracking_velocity = goal
    
    def getTargetPosition(self):
        return self.tracker.get_target_position()
    
    def isDone(self):
        return self.tracker.is_done

    def setTarget(self):
        self.model.setTarget(target_linear_velocity=self.velocity_goal)

    def getAction(self, state, is_deterministic=True):
        self.velocity_vector = self.tracker.getVelocityVector(state["position"][:2])
        self.velocity_goal[0] = self.velocity_vector[0]*self.target_tracking_velocity
        self.velocity_goal[1] = self.velocity_vector[1]*self.target_tracking_velocity
        self.setTarget()
        action = self.model.getAction(state, is_deterministic=is_deterministic)
        return action


class HLControllerFactory:
    def __init__(self):
        self.registered_controllers = {}

    def registerController(self, name, controller):
        self.registered_controllers[name] = controller

    def parseControllerConfiguration(self, cfg: Dict):
        return cfg["hl_task"], cfg["hl_task"]["name"]

    def __call__(self, cfg: Dict, model: Union[RLGamesModel, DiscreteController]):
        new_cfg, mode = self.parseControllerConfiguration(cfg)
        assert mode in list(self.registered_controllers.keys()), "Unknown hl_task mode."
        return self.registered_controllers[mode](model, **new_cfg)


hlControllerFactory = HLControllerFactory()
hlControllerFactory.registerController("position", PositionController)
hlControllerFactory.registerController("pose", PoseController)
hlControllerFactory.registerController("linear_velocity", VelocityTracker)