import matplotlib.pyplot as plt
import numpy as np
import mujoco

class MuJoCoFloatingPlatform:
    """
    A class for the MuJoCo Floating Platform environment."""

    def __init__(self, step_time:float = 0.02, duration:float = 60.0, inv_play_rate:int = 10) -> None:
        """
        Initializes the MuJoCo Floating Platform environment.
        step_time: The time between steps in the simulation.
        duration: The duration of the simulation.
        inv_play_rate: The inverse of rate at which the controller will run.
        
        With a step_time of 0.02, and inv_play_rate of 10, the agent will play every 0.2 seconds. (or 5Hz)
        """

        self.inv_play_rate = inv_play_rate

        self.createModel()
        self.initializeModel()
        self.setupPhysics(step_time, duration)
        self.initForceAnchors()
        self.initializeLoggers()

        self.goal = np.zeros((2), dtype=np.float32)

    def setGoal(self, goal:np.ndarray) -> None:
        """
        Sets the goal for the simulation.
        goal: The goal position for the agent to reach.
        """

        self.goal = goal

    def initializeModel(self) -> None:
        """
        Initializes the mujoco model for the simulation."""

        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "top")

    def setupPhysics(self, step_time: float, duration: float) -> None:
        """
        Sets up the physics parameters for the simulation.
        step_time: The time between steps in the simulation (seconds).
        duration: The duration of the simulation (seconds)."""

        self.model.opt.timestep = step_time
        self.model.opt.gravity = [0,0,0]
        self.duration = duration

    def initializeLoggers(self) -> None:
        """
        Initializes the loggers for the simulation.
        Allowing for the simulation to be replayed/plotted."""

        self.timevals = []
        self.angular_velocity = []
        self.linear_velocity = []
        self.position = []
        self.heading = []

    def createModel(self) -> None:
        """
        A YAML style string that defines the MuJoCo model for the simulation.
        The mass is set to 5.32 kg, the radius is set to 0.31 m.
        The initial position is set to (3, 3, 0.4) m."""

        sphere = """
        <mujoco model="tippe top">
          <option integrator="RK4"/>
        
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
             rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
          </asset>
        
          <worldbody>
            <geom size="10.0 10.0 .01" type="plane" material="grid"/>
            <light pos="0 0 10.0"/>
            <camera name="closeup" pos="0 -3 2" xyaxes="1 0 0 0 1 2"/>
            <body name="top" pos="0 0 .4">
              <freejoint/>
              <geom name="ball" type="sphere" size=".31" mass="5.32"/>
            </body>
          </worldbody>
        
          <keyframe>
            <key name="idle" qpos="3 3 0.4 1 0 0 0" qvel="0 0 0 0 0 0" />
          </keyframe>
        </mujoco>
        """
        self.model = mujoco.MjModel.from_xml_string(sphere)

    def initForceAnchors(self) -> None:
        """"
        Defines where the forces are applied relatively to the center of mass of the body.
        self.forces: 8x3 array of forces, indicating the direction of the force.
        self.positions: 8x3 array of positions, indicating the position of the force."""

        self.forces = np.array([[ 1, -1, 0],
                           [-1,  1, 0],
                           [ 1,  1, 0],
                           [-1, -1, 0],
                           [-1,  1, 0],
                           [ 1, -1, 0],
                           [-1, -1, 0],
                           [ 1,  1, 0]])
        
        self.positions = np.array([[ 1,  1, 0],
                              [ 1,  1, 0],
                              [-1,  1, 0],
                              [-1,  1, 0],
                              [-1, -1, 0],
                              [-1, -1, 0],
                              [ 1, -1, 0],
                              [ 1, -1, 0]]) * 0.2192


    def resetPosition(self) -> None:
        """
        Resets the position of the body to the initial position, (3, 3, 0.4) m"""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def applyForces(self, action) -> None:
        """
        Applies the forces to the body."""

        self.data.qfrc_applied[...] = 0 # Clear applied forces.
        rmat = self.data.xmat[self.body_id].reshape(3,3) # Rotation matrix.
        p = self.data.xpos[self.body_id] # Position of the body.

        # Compute the number of thrusters fired, split the pressure between the nozzles.
        factor = max(np.sum(action), 1) 
        # For each thruster, apply a force if needed.
        for i in range(8):
          # The force applied is the action value (1 or 0), divided by the number of thrusters fired (factor),
          # times the orientation of the force (self.forces), times sqrt(0.5) to normalize the force orientation vector.
          force = action[i] * (1./factor) * self.forces[i] * np.sqrt(0.5)
          # If the force is not zero, apply the force.
          if np.sum(np.abs(force)) > 0:
              force = np.matmul(rmat, force) # Rotate the force to the body frame.
              p2 = np.matmul(rmat, self.positions[i]) + p # Compute the position of the force.
              mujoco.mj_applyFT(self.model, self.data, force, [0,0,0], p2, self.body_id, self.data.qfrc_applied) # Apply the force.

    def updateLoggers(self) -> None:
        """
        Updates the loggers with the current state of the simulation."""

        self.timevals.append(self.data.time)
        self.angular_velocity.append(self.data.qvel[3:6].copy())
        self.linear_velocity.append(self.data.qvel[0:3].copy())
        self.position.append(self.data.qpos[0:3].copy())

    def updateState(self) -> list:
        """
        Updates the state of the simulation."""

        qpos = self.data.qpos.copy() # Copy the pose of the object.
        # Cast the quaternion to the yaw (roll and pitch are invariant).
        siny_cosp = 2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5])
        cosy_cosp = 1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6])
        # Gets the angular and linear velocity.
        linear_velocity = self.data.qvel[0:2].copy() # X and Y velocities.
        angular_velocity = self.data.qvel[5].copy() # Yaw velocity.
        # Returns the state.
        state = {"orientation": [cosy_cosp, siny_cosp], "position": qpos[:2], "linear_velocity": linear_velocity, "angular_velocity": angular_velocity}
        return state

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

        ax[0].plot(self.timevals, self.angular_velocity)
        ax[0].set_title('angular velocity')
        ax[0].set_ylabel('radians / second')

        ax[1].plot(self.timevals, self.linear_velocity)
        ax[1].set_xlabel('time (seconds)')
        ax[1].set_ylabel('meters / second')
        _ = ax[1].set_title('linear_velocity')
        if save:
            fig.savefig("test_velocities.png")

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        ax[0].plot(self.timevals, np.abs(self.position))
        ax[0].set_xlabel('time (seconds)')
        ax[0].set_ylabel('meters')
        _ = ax[0].set_title('position')
        ax[0].set_yscale('log')


        ax[1].plot(np.array(self.position)[:,0], np.array(self.position)[:,1])
        ax[1].set_xlabel('meters')
        ax[1].set_ylabel('meters')
        _ = ax[1].set_title('x y coordinates')
        plt.tight_layout()
        if save:
            fig.savefig("test_positions.png")