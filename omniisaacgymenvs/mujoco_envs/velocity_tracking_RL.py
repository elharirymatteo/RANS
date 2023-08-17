import matplotlib.pyplot as plt
import numpy as np
import mujoco
import torch 

from omniisaacgymenvs.mujoco_envs.mujoco_base_env_RL import MuJoCoFloatingPlatform
from omniisaacgymenvs.mujoco_envs.RL_games_model_4_mujoco import RLGamesModel

class MuJoCoVelTracking(MuJoCoFloatingPlatform):
    def __init__(self, step_time:float = 0.02, duration:float = 240.0, inv_play_rate:int = 10) -> None:
        super().__init__(step_time, duration, inv_play_rate)

    def initializeLoggers(self) -> None:
        super().initializeLoggers()
        self.velocity_goal = []
        self.position_target = []

    def updateLoggers(self, goal, target) -> None:
        super().updateLoggers()
        self.velocity_goal.append(goal)
        self.position_target.append(target)
    

    def runLoop(self, model, xy: np.ndarray) -> None:
        """
        Runs the simulation loop.
        model: the agent.
        xy: 2D position of the body."""

        self.resetPosition() # Resets the position of the body.
        self.data.qpos[:2] = xy # Sets the position of the body.

        while self.duration > self.data.time:
            state = self.updateState() # Updates the state of the simulation.
            # Get the actions from the controller
            action = model.getAction(state)
            # Plays only once every self.inv_play_rate steps.
            for _ in range(self.inv_play_rate):
                self.applyForces(action)
                mujoco.mj_step(self.model, self.data)
                self.updateLoggers(model.get_goal(), model.get_target_position())
    
    def plotSimulation(self, dpi:int = 90, width:int = 1000, height:int = 1000, save:bool = False) -> None:
        """
        Plots the simulation."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

        ax[0].plot(self.timevals, self.angular_velocity)
        ax[0].set_title('angular velocity')
        ax[0].set_ylabel('radians / second')

        ax[1].plot(self.timevals, self.linear_velocity, label="system velocities")
        ax[1].plot(self.timevals, self.velocity_goal, label="target velocities")
        ax[1].legend()
        ax[1].set_xlabel('time (seconds)')
        ax[1].set_ylabel('meters / second')
        _ = ax[1].set_title('linear_velocity')
        if save:
            fig.savefig("test_velocities.png")

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.plot(np.array(self.position_target)[:,0], np.array(self.position_target)[:,1], label="trajectory")
        ax.plot(np.array(self.position)[:,0], np.array(self.position)[:,1], label="system position")
        ax.legend()
        ax.set_xlabel('meters')
        ax.set_ylabel('meters')
        ax.axis("equal")
        _ = ax.set_title('x y coordinates')
        plt.tight_layout()
        if save:
            fig.savefig("test_positions.png")

class TrajectoryTracker:
    def __init__(self, lookahead=0.15, closed=False):
        self.current_point = -1
        self.lookhead = lookahead
        self.closed = closed

    def generateCircle(self, radius=2, num_points=360*10):
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=(not self.closed))
        self.positions = np.array([np.cos(theta) * radius, np.sin(theta) * radius]).T
        self.angles = np.array([-np.sin(theta), np.cos(theta)]).T

    def generateSpiral(self, start_radius=0.5, end_radius=2, num_loop=5, num_points=360*20):
        radius = np.linspace(start_radius, end_radius, num_points, endpoint=(not self.closed))
        theta = np.linspace(0, 2*np.pi*num_loop, num_points, endpoint=(not self.closed))
        self.positions = np.array([np.cos(theta) * radius, np.sin(theta) * radius]).T
        self.angles = np.array([-np.sin(theta), np.cos(theta)]).T
    
    def getTrackingPointIdx(self, position):
        distances = np.linalg.norm(self.positions - position, axis=1)
        if self.current_point == -1:
            self.current_point = 0
        else:
            indices = np.where(distances < self.lookhead)[0]
            if len(indices) > 0:
                indices = indices[indices < 60]
                self.current_point = np.max(indices)

    def rollTrajectory(self):
        if self.closed:
            self.positions = np.roll(self.positions, -self.current_point, axis=0)
            self.angles = np.roll(self.angles, -self.current_point, axis=0)
            self.current_point = 0
        else:
            self.positions = self.positions[self.current_point:]
            self.angles = self.angles[self.current_point:]
            self.current_point = 0

    def getPointForTracking(self):
        position = self.positions[self.current_point]
        angle = self.angles[self.current_point]
        self.rollTrajectory()
        return position, angle
    
    def get_target_position(self):
        return self.target_position
    
    def computeVelocityVector(self, target_position, position):
        diff = target_position - position
        return diff / np.linalg.norm(diff)
    
    def getVelocityVector(self, position):
        self.getTrackingPointIdx(position)
        self.target_position, target_angle = self.getPointForTracking()
        velocity_vector = self.computeVelocityVector(self.target_position, position)
        return velocity_vector
    
class VelocityTracker:
    def __init__(self, trajectory_tracker: TrajectoryTracker, model: RLGamesModel, target_tracking_velocity:float = 0.25):
        self.trajectory_tracker = trajectory_tracker
        self.model = model
        self.target_tracking_velocity = target_tracking_velocity

        self.obs_state = torch.zeros((1,10), dtype=torch.float32, device="cuda")
    
    def get_goal(self):
        return self.velocity_vector*self.target_tracking_velocity
    
    def get_target_position(self):
        return self.trajectory_tracker.get_target_position()

    def make_observation_buffer(self, state, velocity_vector):
        self.obs_state[0,:2] = torch.tensor(state["orientation"], dtype=torch.float32, device="cuda")
        self.obs_state[0,2:4] = torch.tensor(state["linear_velocity"], dtype=torch.float32, device="cuda")
        self.obs_state[0,4] = state["angular_velocity"]
        self.obs_state[0,5] = 2
        self.obs_state[0,6:8] = torch.tensor(velocity_vector, dtype=torch.float32, device="cuda")

    def getAction(self, state):
        self.velocity_vector = self.trajectory_tracker.getVelocityVector(state["position"])
        velocity_goal = self.velocity_vector*self.target_tracking_velocity - state["linear_velocity"]
        self.make_observation_buffer(state, velocity_goal)
        action = self.model.getAction(self.obs_state)
        return action

if __name__ == "__main__":

    tracker = TrajectoryTracker(lookahead=0.15, closed=True)
    tracker.generateCircle()
    #tracker = TrajectoryTracker(lookahead=0.15, closed=False)
    #tracker.generateSpiral()

    cfg_path = "/home/antoine/Documents/Orbitals/Omniverse/omniisaacgymenvs/cfg/train/virtual_floating_platform/MFP2D_PPOmulti_dict_MLP.yaml"
    model_path = "/home/antoine/Documents/Orbitals/Omniverse/omniisaacgymenvs/runs/MFP2D_Virtual_TrackXYVelocity/nn/last_MFP2D_Virtual_TrackXYVelocity_ep_1000_rew_637.1514.pth"
    model = RLGamesModel(cfg_path, model_path)

    velocity_tracker = VelocityTracker(tracker, model)

    env = MuJoCoVelTracking()
    env.runLoop(velocity_tracker, [0,0])
    env.plotSimulation(save=True)