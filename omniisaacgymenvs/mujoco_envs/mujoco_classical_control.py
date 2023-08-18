import matplotlib.pyplot as plt
import numpy as np
import mujoco

from omniisaacgymenvs.mujoco_envs.mujoco_base_env import MuJoCoFloatingPlatform

class MujocoClassicalControlEnv(MuJoCoFloatingPlatform):
    """
    A class for the MuJoCo Floating Platform environment."""

    def __init__(self, step_time:float = 0.02, duration:float = 60.0, inv_play_rate:int = 10,
                 mass:float = 5.32, max_thrust:float = 1.0, radius:float = 0.31) -> None:
        """
        Initializes the MuJoCo Floating Platform environment.
        step_time: The time between steps in the simulation.
        duration: The duration of the simulation.
        inv_play_rate: The inverse of rate at which the controller will run.
        
        With a step_time of 0.02, and inv_play_rate of 10, the agent will play every 0.2 seconds. (or 5Hz)
        """ 
        super().__init__(step_time, duration, inv_play_rate, mass, max_thrust, radius)

    def updateState(self) -> list:
        """
        Updates the state of the simulation."""

        qpos = self.data.qpos.copy() # Copy the pose of the object.
        # Cast the quaternion to the yaw (roll and pitch are invariant).
        siny_cosp = 2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5])
        cosy_cosp = 1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6])
        orient_z = np.arctan2(siny_cosp, cosy_cosp)
        # Compute the distance to the goal. (in the global frame)
        dist_to_goal = self.goal - qpos[:2]
        # Gets the angular and linear velocity.
        linear_velocity = self.data.qvel[0:2].copy() # X and Y velocities.
        angular_velocity = self.data.qvel[5].copy() # Yaw velocity.
        # Returns the state.
        return orient_z, dist_to_goal, angular_velocity, linear_velocity

    def runLoop(self, controller, xy: np.ndarray) -> None:
        """
        Runs the simulation loop.
        controller: the position controller.
        xy: 2D position of the body."""

        self.resetPosition() # Resets the position of the body.
        self.data.qpos[:2] = xy # Sets the position of the body.

        while self.duration > self.data.time:
            state = self.updateState() # Updates the state of the simulation.
            # Get the actions from the controller
            action = controller.getAction(state)
            # Plays only once every self.inv_play_rate steps.
            for _ in range(self.inv_play_rate):
                self.applyForces(action)
                mujoco.mj_step(self.model, self.data)
                self.updateLoggers()

if __name__ == "__main__":

    model = SOMETHING
    env = MuJoCoFloatingPlatform()
    env.runLoop(model, [3,0])
    env.plotSimulation(save=False)