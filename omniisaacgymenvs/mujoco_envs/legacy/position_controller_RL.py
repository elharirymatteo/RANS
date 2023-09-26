from typing import Callable, NamedTuple, Optional, Union, List
import matplotlib.pyplot as plt
import numpy as np
import argparse
import mujoco
import torch
import os

from omniisaacgymenvs.mujoco_envs.environments.mujoco_base_env import MuJoCoFloatingPlatform
from omniisaacgymenvs.mujoco_envs.controllers.RL_games_model_4_mujoco import RLGamesModel

class MuJoCoPositionControl(MuJoCoFloatingPlatform):
    def __init__(self, step_time:float = 0.02, duration:float = 60.0, inv_play_rate:int = 10,
                 mass:float = 5.32, max_thrust:float = 1.0, radius:float = 0.31) -> None:
        super().__init__(step_time, duration, inv_play_rate, mass, max_thrust, radius)

    def initializeLoggers(self) -> None:
        super().initializeLoggers()
        self.logs["position_target"] = []

    def updateLoggers(self, target) -> None:
        super().updateLoggers()
        self.logs["position_target"].append(target)

    def applyFriction(self, fdyn=0.1, fstat=0.1, tdyn=0.05, tstat=0.0):
        lin_vel = self.data.qvel[:3]
        lin_vel_norm = np.linalg.norm(lin_vel)
        ang_vel = self.data.qvel[-1]
        forces = self.data.qfrc_applied[:3]
        forces_norm = np.linalg.norm(forces)
        torques = self.data.qfrc_applied[3:]
        torques_norm = np.linalg.norm(torques)
        #if (forces_norm > fstat) or (torques_norm > tstat):
        if lin_vel_norm > 0.001:
            lin_vel_normed = np.array(lin_vel) / lin_vel_norm
            force = -lin_vel_normed * fdyn
            force[-1] = 0
            mujoco.mj_applyFT(self.model, self.data, list(force), [0,0,0], self.data.qpos[:3], self.body_id, self.data.qfrc_applied)
        if ang_vel > 0.001:
            torque = - np.sign(ang_vel) * tdyn
            mujoco.mj_applyFT(self.model, self.data, [0,0,0], [0,0,torque], self.data.qpos[:3], self.body_id, self.data.qfrc_applied)
        #else:
        #    self.data.qfrc_applied[:3] = 0

    def runLoop(self, model, xy: np.ndarray) -> None:
        """
        Runs the simulation loop.
        model: the agent.
        xy: 2D position of the body."""

        self.resetPosition() # Resets the position of the body.
        self.data.qpos[:2] = xy # Sets the position of the body.

        while (self.duration > self.data.time) and (model.isDone() == False):
            state = self.updateState() # Updates the state of the simulation.
            # Get the actions from the controller
            self.actions = model.getAction(state)
            # Plays only once every self.inv_play_rate steps.
            for _ in range(self.inv_play_rate):
                self.applyForces(self.actions)
                mujoco.mj_step(self.model, self.data)
                self.updateLoggers(model.getGoal())

    def plotSimulation(self, dpi:int = 90, width:int = 1000, height:int = 1000, save:bool = True, save_dir:str = "position_exp") -> None:
        """
        Plots the simulation."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

        ax[0].plot(self.logs["timevals"], self.logs["angular_velocity"])
        ax[0].set_title('angular velocity')
        ax[0].set_ylabel('radians / second')

        ax[1].plot(self.logs["timevals"], self.logs["linear_velocity"], label="system velocities")
        ax[1].legend()
        ax[1].set_xlabel('time (seconds)')
        ax[1].set_ylabel('meters / second')
        _ = ax[1].set_title('linear_velocity')
        if save:
            try:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, "velocities.png"))
            except Exception as e:
                print("Saving failed: ", e)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.scatter(np.array(self.logs["position_target"])[:,0], np.array(self.logs["position_target"])[:,1], label="position goals")
        ax.plot(np.array(self.logs["position"])[:,0], np.array(self.logs["position"])[:,1], label="system position")
        ax.legend()
        ax.set_xlabel('meters')
        ax.set_ylabel('meters')
        ax.axis("equal")
        _ = ax.set_title('x y coordinates')
        plt.tight_layout()
        if save:
            try:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, "positions.png"))
            except Exception as e:
                print("Saving failed: ", e)

class PositionController:
    def __init__(self, model: RLGamesModel, goal_x: List[float], goal_y: List[float], distance_threshold: float = 0.03) -> None:
        self.model = model
        self.goals = np.array([goal_x, goal_y]).T
        self.current_goal = self.goals[0]
        self.distance_threshold = distance_threshold

        self.obs_state = torch.zeros((1,10), dtype=torch.float32, device="cuda")

    def isGoalReached(self, state):
        dist = np.linalg.norm(self.current_goal - state["position"])
        if dist < self.distance_threshold:
            return True
    
    def getGoal(self):
        return self.current_goal
    
    def setGoal(self, goal):
        self.current_goal = goal
        self.goals = np.array([goal])

    def isDone(self):
        return len(self.goals) == 0
    
    def getObs(self):
        return self.obs_state.cpu().numpy()

    def makeObservationBuffer(self, state):
        self.obs_state[0,:2] = torch.tensor(state["orientation"], dtype=torch.float32, device="cuda")
        self.obs_state[0,2:4] = torch.tensor(state["linear_velocity"], dtype=torch.float32, device="cuda")
        self.obs_state[0,4] = state["angular_velocity"]
        self.obs_state[0,5] = 0
        self.obs_state[0,6:8] = torch.tensor(self.current_goal - state["position"], dtype=torch.float32, device="cuda")

    def getAction(self, state, is_deterministic: bool = True):
        if self.isGoalReached(state):
            print("Goal reached!")
            if len(self.goals) > 1:
                self.current_goal = self.goals[1]
                self.goals = self.goals[1:]
            else:
                self.goals = []
        self.makeObservationBuffer(state)
        return self.model.getAction(self.obs_state, is_deterministic=is_deterministic)

def parseArgs():
    parser = argparse.ArgumentParser("Generates meshes out of Digital Elevation Models (DEMs) or Heightmaps.")
    parser.add_argument("--model_path", type=str, default=None, help="The path to the model to be loaded. It must be a velocity tracking model.")
    parser.add_argument("--config_path", type=str, default=None, help="The path to the network configuration to be loaded.")
    parser.add_argument("--goal_x", type=float, nargs="+", default=None, help="List of x coordinates for the goals to be reached by the platform. In world frame, meters.")
    parser.add_argument("--goal_y", type=float, nargs="+", default=None, help="List of y coordinates for the goals to be reached by the platform. In world frame, meters.")
    parser.add_argument("--sim_duration", type=float, default=240, help="The length of the simulation. In seconds.")
    parser.add_argument("--play_rate", type=float, default=5.0, help="The frequency at which the agent will played. In Hz. Note, that this depends on the sim_rate, the agent my not be able to play at this rate depending on the sim_rate value. To be consise, the agent will play at: sim_rate / int(sim_rate/play_rate)")
    parser.add_argument("--sim_rate", type=float, default=50.0, help="The frequency at which the simulation will run. In Hz.")
    parser.add_argument("--save_dir", type=str, default="position_exp", help="The path to the folder in which the results will be stored.")
    parser.add_argument("--platform_mass", type=float, default=5.32, help="The mass of the floating platform. In Kg.")
    parser.add_argument("--platform_radius", type=float, default=0.31, help="The radius of the floating platform. In meters.")
    parser.add_argument("--platform_max_thrust", type=float, default=1.0, help="The maximum thrust of the floating platform. In newtons.")
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

if __name__ == "__main__":
    # Collects args
    args, _ = parseArgs()
    # Checks args
    assert os.path.exists(args.model_path), "The model file does not exist."
    assert os.path.exists(args.config_path), "The configuration file does not exist."
    assert not args.goal_x is None, "The x coordinates of the goals must be specified."
    assert not args.goal_y is None, "The y coordinates of the goals must be specified."
    assert args.sim_rate > args.play_rate, "The simulation rate must be greater than the play rate."
    assert args.sim_duration > 0, "The simulation duration must be greater than 0."
    assert args.play_rate > 0, "The play rate must be greater than 0."
    assert args.sim_rate > 0, "The simulation rate must be greater than 0."
    assert args.platform_mass > 0, "The mass of the platform must be greater than 0."
    assert args.platform_radius > 0, "The radius of the platform must be greater than 0."
    assert args.platform_max_thrust > 0, "The maximum thrust of the platform must be greater than 0."
    assert len(args.goal_x) == len(args.goal_y), "The number of x coordinates must be equal to the number of y coordinates."
    # Try to create the save directory
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except:
        raise ValueError("Could not create the save directory.")
    # Instantiates the RL agent
    model = RLGamesModel(args.config_path, args.model_path)
    #  Creates the velocity tracker
    position_controller = PositionController(model, args.goal_x, args.goal_y)
    # Creates the environment
    env = MuJoCoPositionControl(step_time=1.0/args.sim_rate, duration=args.sim_duration, inv_play_rate=int(args.sim_rate/args.play_rate),
                            mass=args.platform_mass, radius=args.platform_radius, max_thrust=args.platform_max_thrust)
    # Runs the simulation
    env.runLoop(position_controller, [0,0])
    # Plots the simulation
    env.plotSimulation(save_dir = args.save_dir)
    # Saves the simulation data
    env.saveSimulationData(save_dir = args.save_dir)