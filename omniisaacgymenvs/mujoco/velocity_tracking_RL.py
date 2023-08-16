import matplotlib.pyplot as plt
import numpy as np
import mujoco
import torch 

from omniisaacgymenvs.mujoco.mujoco_base_env_RL import MujocoBaseRL
from omniisaacgymenvs.mujoco.RL_games_model_4_mujoco import RLGamesModel

class MuJoCoVelTracking(MujocoBaseRL):
    def __init__(self, step_time:float = 0.02, duration:float = 60.0, inv_play_rate:int = 10) -> None:
        super().__init__(step_time, duration, inv_play_rate)

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
                self.updateLoggers()
    
    def plotSimulation(self, dpi:int = 120, width:int = 600, height:int = 800, save:bool = False) -> None:
        """
        Plots the simulation."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

        ax[0].plot(env.timevals, env.angular_velocity)
        ax[0].set_title('angular velocity')
        ax[0].set_ylabel('radians / second')

        ax[1].plot(env.timevals, env.linear_velocity)
        ax[1].set_xlabel('time (seconds)')
        ax[1].set_ylabel('meters / second')
        _ = ax[1].set_title('linear_velocity')
        if save:
            fig.savefig("test_velocities.png")

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        ax[0].plot(env.timevals, np.abs(env.position))
        ax[0].set_xlabel('time (seconds)')
        ax[0].set_ylabel('meters')
        _ = ax[0].set_title('position')
        ax[0].set_yscale('log')


        ax[1].plot(np.array(env.position)[:,0], np.array(env.position)[:,1])
        ax[1].set_xlabel('meters')
        ax[1].set_ylabel('meters')
        _ = ax[1].set_title('x y coordinates')
        plt.tight_layout()
        if save:
            fig.savefig("test_positions.png")

class TrajectoryTracker:
    def __init__(self):
        self.current_point = -1
        self.lookhead = 0.3
        self.closed = True

    def generateCircle(self, radius=2, num_points=1080):
        theta = np.linspace(0, 2*np.pi, num_points, endpoint=(not self.closed))
        self.positions = np.array([np.cos(theta) * radius, np.sin(theta) * radius]).T
        self.angles = np.array([-np.sin(theta), np.cos(theta)]).T
    
    def getTrackingPointIdx(self, position):
        distances = np.linalg.norm(self.positions - position, axis=1)
        if self.current_point == -1:
            self.current_point = 0
        else:
            indices = np.argwhere(distances < self.lookhead)[0]
            self.current_point = np.max(indices)

    def rollTrajectory(self):
        if self.closed:
            self.positions = np.roll(self.positions, -self.current_point, axis=0)
            self.angles = np.roll(self.angles, -self.current_point, axis=0)
            self.current_point = 0
        else:
            self.positions = self.positions[self.current_point:]
            self.angles = self.angles[self.current_point:]

    def getPointForTracking(self):
        position = self.positions[self.current_point]
        angle = self.angles[self.current_point]
        self.rollTrajectory()
        return position, angle
    
    def computeVelocityVector(self, target_position, position):
        diff = target_position - position
        return diff / np.linalg.norm(diff)
    
    def getVelocityVector(self, position):
        self.getTrackingPointIdx(position)
        target_position, target_angle = self.getPointForTracking()
        velocity_vector = self.computeVelocityVector(target_position, position)
        return velocity_vector
    
class VelocityTracker:
    def __init__(self, trajectory_tracker: TrajectoryTracker, model: RLGamesModel, target_tracking_velocity:float = 0.35):
        self.trajectory_tracker = trajectory_tracker
        self.model = model
        self.target_tracking_velocity = target_tracking_velocity

        self.obs_state = torch.zeros((1,10), dtype=torch.float32, device="cuda")

    def make_observation_buffer(self, state, velocity_vector):
        self.obs_state[0,:2] = torch.tensort(state["heading"], dtype=torch.float32, device="cuda")
        self.obs_state[0,2:4] = torch.tensor(state["linear_velocity"], dtype=torch.float32, device="cuda")
        self.obs_state[0,4] = state["angular_velocity"]
        self.obs_state[0,5] = 2
        self.obs_state[0,6:8] = torch.tensor(velocity_vector, dtype=torch.float32, device="cuda")

    def getAction(self, state):
        velocity_vector = self.trajectory_tracker.getVelocityVector(state["position"])
        veliocity_goal = velocity_vector*self.target_tracking_velocity
        self.make_observation_buffer(state, veliocity_goal)
        action = self.model.getAction(self.obs_state)
        return action

if __name__ == "__main__":

    tracker = TrajectoryTracker()
    tracker.generateCircle()

    model = RLGamesModel()
    model.loadConfig("/home/antoine/Documents/Orbitals/Omniverse/omniisaacgymenvs/cfg/train/virtual_floating_platform/MFP2D_PPOmulti_dict_MLP.yaml")
    model.restore("/home/antoine/Documents/Orbitals/Omniverse/omniisaacgymenvs/runs/MFP2D_Virtual_TrackXYVelocity/nn/last_MFP2D_Virtual_TrackXYVelocity_ep_1000_rew_634.26105.pth")

    env = MuJoCoVelTracking()
    env.runLoop(model, [0,0])
    env.plotSimulation(save=False)