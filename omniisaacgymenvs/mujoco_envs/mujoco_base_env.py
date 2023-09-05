from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mujoco
import math
import os

default_cfg = {}
default_cfg["seed"] = 42
default_cfg['disturbances'] = {}
default_cfg['disturbances']['use_uneven_floor'] = True
default_cfg['disturbances']['use_sinusoidal_floor'] = False
default_cfg['disturbances']['floor_min_freq'] = 0.25
default_cfg['disturbances']['floor_max_freq'] = 3.0
default_cfg['disturbances']['floor_min_offset'] = -6.0
default_cfg['disturbances']['floor_max_offset'] = 6.0
default_cfg['disturbances']['max_floor_force'] = 0.5
default_cfg['disturbances']['min_floor_force'] = 0.5
default_cfg['disturbances']['use_torque_diturbance'] = True
default_cfg['disturbances']['use_sinusoidal_torque'] = False
default_cfg['disturbances']['max_torque'] = 0.1
default_cfg['disturbances']['min_torque'] = 0.1
default_cfg['disturbances']['add_noise_on_pos'] = True
default_cfg['disturbances']['position_noise_min'] = -0.01
default_cfg['disturbances']['position_noise_max'] = 0.01
default_cfg['disturbances']['add_noise_on_vel'] = True
default_cfg['disturbances']['velocity_noise_min'] = -0.01
default_cfg['disturbances']['velocity_noise_max'] = 0.01
default_cfg['disturbances']['add_noise_on_heading'] = True
default_cfg['disturbances']['heading_noise_min'] = -0.025
default_cfg['disturbances']['heading_noise_max'] = 0.025
default_cfg['disturbances']['add_noise_on_act'] = True
default_cfg['disturbances']['min_action_noise'] = -0.05
default_cfg['disturbances']['max_action_noise'] = 0.05
default_cfg['disturbances']['num_thrusters_to_kill'] = 0
default_cfg["max_spawn_dist"] = 4.0
default_cfg["min_spawn_dist"] = 3.0
default_cfg["kill_dist"] = 6.0

class RandomSpawn:
    def __init__(self, cfg):
        self._rng = np.random.default_rng()
        self._max_spawn_dist = cfg["max_spawn_dist"]
        self._min_spawn_dist = cfg["min_spawn_dist"]
        self._kill_dist = cfg["kill_dist"]

    def getInitialCondition(self):
        theta = self._rng.uniform(-np.pi, np.pi, 1)
        r = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)
        initial_position = [np.cos(theta) * r, np.sin(theta) * r]
        heading = self._rng.uniform(-np.pi, np.pi, 1)
        initial_orientation = [np.cos(heading*0.5), 0, 0, np.sin(heading*0.5)]
        return initial_position, initial_orientation

class RandomKillThrusters:
    """
    Randomly kills thrusters."""

    def __init__(self, cfg):
        self._rng = np.random.default_rng()
        self._num_thrusters_to_kill = cfg['disturbances']['num_thrusters_to_kill']
        self.killed_thrusters_id = []

    def generate_thruster_kills(self):
        self.killed_thrusters_id = self._rng.choice(8, self._num_thrusters_to_kill, replace=False) #[2,3]
        print("Killed thrusters: ", self.killed_thrusters_id)
        
class UnevenFloorDisturbance:
    """
    Creates disturbances on the platform by simulating an uneven floor."""

    def __init__(self, cfg: dict) -> None:
        # Uneven floor generation
        self._rng = np.random.default_rng(cfg["seed"])
        self._use_uneven_floor = cfg['disturbances']['use_uneven_floor']
        self._use_sinusoidal_floor = cfg['disturbances']['use_sinusoidal_floor']
        self._min_freq = cfg['disturbances']['floor_min_freq']
        self._max_freq = cfg['disturbances']['floor_max_freq']
        self._min_offset = cfg['disturbances']['floor_min_offset']
        self._max_offset = cfg['disturbances']['floor_max_offset']
        self._max_floor_force = cfg['disturbances']['max_floor_force'] 
        self._min_floor_force = cfg['disturbances']['min_floor_force'] 
        self._max_floor_force = math.sqrt(self._max_floor_force**2 / 2)
        self._min_floor_force = math.sqrt(self._min_floor_force**2 / 2)

        self._floor_forces = np.zeros(3, dtype=np.float32)
        self._floor_x_freq = 0
        self._floor_y_freq = 0
        self._floor_x_offset = 0
        self._floor_y_offset = 0

    def generate_floor(self) -> None:
        """
        Generates the uneven floor."""
        if self._use_uneven_floor:
            if self._use_sinusoidal_floor:
                self._floor_x_freq   = self._rng.uniform(self._min_freq, self._max_freq, 1)
                self._floor_y_freq   = self._rng.uniform(self._min_freq, self._max_freq, 1)
                self._floor_x_offset = self._rng.uniform(self._min_offset, self._max_offset, 1)
                self._floor_y_offset = self._rng.uniform(self._min_offset, self._max_offset, 1)
            else:
                r = self._rng.uniform(self._min_floor_force, self._max_floor_force, 1)
                theta = self._rng.uniform(0, 1, 1) * math.pi * 2
                self._floor_forces[0] = np.cos(theta) * r
                self._floor_forces[1] = np.sin(theta) * r

    def get_floor_forces(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the floor forces for the current state of the robot."""
        if self._use_uneven_floor:
            if self._use_sinusoidal_floor:
                self._floor_forces[0] = np.sin(root_pos[0] * self._floor_x_freq + self._floor_x_offset) * self._max_floor_force
                self._floor_forces[1] = np.sin(root_pos[1] * self._floor_y_freq + self._floor_y_offset) * self._max_floor_force
       
        return self._floor_forces


class TorqueDisturbance:
    """
    Creates disturbances on the platform by simulating a torque applied to its center."""

    def __init__(self, cfg: dict) -> None:
        self._rng = np.random.default_rng(cfg["seed"])
        # Uneven floor generation
        self._use_torque_disturbance = cfg['disturbances']['use_torque_diturbance']
        self._use_sinusoidal_torque = cfg['disturbances']['use_sinusoidal_torque']
        self._max_torque = cfg['disturbances']['max_torque']
        self._min_torque = cfg['disturbances']['min_torque']

        # use the same min/max frequencies and offsets for the floor
        self._min_freq = cfg['disturbances']['floor_min_freq']
        self._max_freq = cfg['disturbances']['floor_max_freq']
        self._min_offset = cfg['disturbances']['floor_min_offset']
        self._max_offset = cfg['disturbances']['floor_max_offset']

        self._torque_forces = np.zeros(3, dtype=np.float32)
        self._torque_freq = 0
        self._torque_offset = 0

    def generate_torque(self) -> None:
        """
        Generates the torque disturbance."""
        if self._use_torque_disturbance:
            if self._use_sinusoidal_torque:
                #  use the same min/max frequencies and offsets for the floor
                self._torque_freq = self._rng.uniform(self._min_freq, self._max_freq, 1)
                self._torque_offset = self._rng.uniform(self._min_offset, self._max_offset, 1)
            else:
                r = self._rng.uniform(self._min_torque, self._max_torque, 1) * self._rng.choice([1,-1])
                self._torque_forces[2] = r

    def get_torque_disturbance(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the torque for the current state of the robot."""
        if self._use_torque_disturbance:
            if self._use_sinusoidal_torque:
                self._torque_forces[2] = np.sin(root_pos * self._torque_freq + self._torque_offset) * self._max_torque

        return self._torque_forces


class NoisyObservations:
    """
    Adds noise to the observations of the robot."""

    def __init__(self, cfg: dict) -> None:
        self._rng = np.random.default_rng(seed=cfg['seed'])
        self._add_noise_on_pos = cfg['disturbances']['add_noise_on_pos']
        self._position_noise_min = cfg['disturbances']['position_noise_min']
        self._position_noise_max = cfg['disturbances']['position_noise_max']
        self._add_noise_on_vel = cfg['disturbances']['add_noise_on_vel']
        self._velocity_noise_min = cfg['disturbances']['velocity_noise_min']
        self._velocity_noise_max = cfg['disturbances']['velocity_noise_max']
        self._add_noise_on_heading = cfg['disturbances']['add_noise_on_heading']
        self._heading_noise_min = cfg['disturbances']['heading_noise_min']
        self._heading_noise_max = cfg['disturbances']['heading_noise_max']
    
    def add_noise_on_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Adds noise to the position of the robot."""

        if self._add_noise_on_pos:
            pos += self._rng.uniform(self._position_noise_min, self._position_noise_max, pos.shape)
        return pos
    
    def add_noise_on_vel(self, vel: np.ndarray) -> np.ndarray:
        """
        Adds noise to the velocity of the robot."""

        if self._add_noise_on_vel:
            vel += self._rng.uniform(self._velocity_noise_min, self._velocity_noise_max, vel.shape)
        return vel
    
    def add_noise_on_heading(self, heading: np.ndarray) -> np.ndarray:
        """
        Adds noise to the heading of the robot."""

        if self._add_noise_on_heading:
            heading += self._rng.uniform(self._heading_noise_min, self._heading_noise_max, heading.shape)
        return heading


class NoisyActions:
    """
    Adds noise to the actions of the robot."""

    def __init__(self, cfg: dict) -> None:
        self._rng = np.random.default_rng(seed=cfg['seed'])
        self._add_noise_on_act = cfg['disturbances']['add_noise_on_act']
        self._min_action_noise = cfg['disturbances']['min_action_noise']
        self._max_action_noise = cfg['disturbances']['max_action_noise']

    def add_noise_on_act(self, act: np.ndarray) -> np.ndarray:
        """
        Adds noise to the actions of the robot."""

        if self._add_noise_on_act:
            act += self._rng.uniform(self._min_action_noise, self._max_action_noise, 1)
        return act
    

class MuJoCoFloatingPlatform:
    """
    A class for the MuJoCo Floating Platform environment."""

    def __init__(self, step_time:float = 0.02, duration:float = 60.0, inv_play_rate:int = 10,
                 mass:float = 5.32, max_thrust:float = 1.0, radius:float = 0.31, cfg:dict=default_cfg) -> None:
        """
        Initializes the MuJoCo Floating Platform environment.
        step_time: The time between steps in the simulation.
        duration: The duration of the simulation.
        inv_play_rate: The inverse of rate at which the controller will run.
        
        With a step_time of 0.02, and inv_play_rate of 10, the agent will play every 0.2 seconds. (or 5Hz)
        """

        self.inv_play_rate = inv_play_rate
        self.mass = mass
        self.max_thrust = max_thrust
        self.radius = radius

        self.AN = NoisyActions(cfg)
        self.ON = NoisyObservations(cfg)
        self.TD = TorqueDisturbance(cfg)
        self.UF = UnevenFloorDisturbance(cfg)
        self.TK = RandomKillThrusters(cfg)
        self.RS = RandomSpawn(cfg)

        self.createModel()
        self.initializeModel()
        self.setupPhysics(step_time, duration)
        self.initForceAnchors()
        self.initializeLoggers()

        self.reset()
        self.csv_datas = []

    def reset(self, initial_position=[0,0,0], initial_orientation=[1,0,0,0]):
        """
        Resets the simulation."""

        self.resetPosition(initial_position=initial_position, initial_orientation=initial_orientation)
        self.initializeLoggers()
        self.UF.generate_floor()
        self.TD.generate_torque()
        self.TK.generate_thruster_kills()

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

        self.logs = {}
        self.logs["timevals"] = []
        self.logs["angular_velocity"] = []
        self.logs["linear_velocity"] = []
        self.logs["position"] = []
        self.logs["quaternion"] = []
        self.logs["actions"] = []

    def createModel(self) -> None:
        """
        A YAML style string that defines the MuJoCo model for the simulation.
        The mass is set to 5.32 kg, the radius is set to 0.31 m.
        The initial position is set to (3, 3, 0.4) m."""

        sphere_p1 = """
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
        """
        sphere_p2 = '<geom name="ball" type="sphere" size="'+str(self.radius)+'" mass="'+str(self.mass)+'"/>'
        sphere_p3 = """
            </body>
          </worldbody>
          <keyframe>
            <key name="idle" qpos="3 3 0.4 1 0 0 0" qvel="0 0 0 0 0 0" />
          </keyframe>
        </mujoco>
        """
        sphere = "\n".join([sphere_p1, sphere_p2, sphere_p3])
        
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
        # Normalize the forces.
        self.forces = self.forces / np.linalg.norm(self.forces, axis=1).reshape(-1, 1)
        # Multiply by the max thrust.
        self.forces = self.forces * self.max_thrust
        
        self.positions = np.array([[ 1,  1, 0],
                              [ 1,  1, 0],
                              [-1,  1, 0],
                              [-1,  1, 0],
                              [-1, -1, 0],
                              [-1, -1, 0],
                              [ 1, -1, 0],
                              [ 1, -1, 0]]) * 0.2192

    def resetPosition(self, initial_position=[0,0], initial_orientation=[1,0,0,0]) -> None:
        """
        Resets the position of the body and sets its velocity to 0.
        Resets the timer as well."""

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:2] = initial_position[:2]
        self.data.qpos[3:7] = initial_orientation
        self.data.qvel = 0

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
            if self.TK.killed_thrusters_id is not None and i in self.TK.killed_thrusters_id:
                continue
            # The force applied is the action value (1 or 0), divided by the number of thrusters fired (factor),
            force = self.AN.add_noise_on_act(action[i])
            force = force * (1./factor) * self.forces[i]
            # If the force is not zero, apply the force.
            if np.sum(np.abs(force)) > 0:
                force = np.matmul(rmat, force) # Rotate the force to the global frame.
                p2 = np.matmul(rmat, self.positions[i]) + p # Compute the position of the force.
                mujoco.mj_applyFT(self.model, self.data, force, [0,0,0], p2, self.body_id, self.data.qfrc_applied) # Apply the force.

        uf_forces = self.UF.get_floor_forces(self.data.qpos[:2])#/self.inv_play_rate
        td_forces = self.TD.get_torque_disturbance(self.data.qpos[:2])#/self.inv_play_rate
        mujoco.mj_applyFT(self.model, self.data, uf_forces, td_forces, self.data.qpos[:3], self.body_id, self.data.qfrc_applied) # Apply the force.

    def updateLoggers(self) -> None:
        """
        Updates the loggers with the current state of the simulation."""

        self.logs["timevals"].append(self.data.time)
        self.logs["angular_velocity"].append(self.data.qvel[3:6].copy())
        self.logs["linear_velocity"].append(self.data.qvel[0:3].copy())
        self.logs["position"].append(self.data.qpos[0:3].copy())
        self.logs["quaternion"].append(np.roll(self.data.qpos[3:].copy(),-1))
        self.logs["actions"].append(self.actions)

    def updateState(self) -> Dict[str, np.ndarray]:
        """
        Updates the state of the simulation."""

        qpos = self.data.qpos.copy() # Copy the pose of the object.
        pos = qpos[:2]
        pos = self.ON.add_noise_on_pos(qpos[:2])
        # Cast the quaternion to the yaw (roll and pitch are invariant).
        # Compute the heading
        siny_cosp = 2 * (qpos[3] * qpos[6] + qpos[4] * qpos[5])
        cosy_cosp = 1 - 2 * (qpos[5] * qpos[5] + qpos[6] * qpos[6])
        orient_z = np.arctan2(siny_cosp, cosy_cosp)
        orient_z = self.ON.add_noise_on_heading(orient_z) 
        siny_cosp = np.sin(orient_z)
        cosy_cosp = np.cos(orient_z)
        # Gets the angular and linear velocity.
        linear_velocity = self.data.qvel[0:2].copy() # X and Y velocities.
        linear_velocity = self.ON.add_noise_on_vel(linear_velocity)
        angular_velocity = self.data.qvel[5].copy() # Yaw velocity.
        angular_velocity = self.ON.add_noise_on_vel(angular_velocity)
        # Returns the state.
        state = {"orientation": [cosy_cosp, siny_cosp], "position": pos, "linear_velocity": linear_velocity, "angular_velocity": angular_velocity}
        return state

    def runLoop(self, model, initial_position=[0,0], initial_orientation=[1,0,0,0]) -> None:
        """
        Runs the simulation loop.
        model: the agent.
        xy: 2D position of the body."""

        self.reset(initial_position=initial_position, initial_orientation=initial_orientation)

        while self.duration > self.data.time:
            state = self.updateState() # Updates the state of the simulation.
            # Get the actions from the controller
            action = model.getAction(state)
            # Plays only once every self.inv_play_rate steps.
            for _ in range(self.inv_play_rate):
                self.applyForces(action)
                mujoco.mj_step(self.model, self.data)
                self.updateLoggers()

    def runLoopForNSteps(self, model, initial_position: np.ndarray=[0,0,0], initial_orientation:np.ndarray=[1,0,0,0], max_steps=502) -> None:
        """
        Runs the simulation loop.
        model: the agent.
        xy: 2D position of the body."""

        self.resetPosition(initial_position=initial_position, initial_orientation=initial_orientation) # Resets the position of the body.
        i = 0
        while i < max_steps:
            state = self.updateState() # Updates the state of the simulation.
            # Get the actions from the controller
            action = model.getAction(state)
            # Plays only once every self.inv_play_rate steps.
            for _ in range(self.inv_play_rate):
                self.applyForces(action)
                mujoco.mj_step(self.model, self.data)
                self.updateLoggers()
            i += 1
    
    def plotSimulation(self, dpi:int = 120, width:int = 600, height:int = 800, save:bool = False, save_dir:str = "mujoco_experiment") -> None:
        """
        Plots the simulation."""

        figsize = (width / dpi, height / dpi)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

        ax[0].plot(self.logs["timevals"], self.logs["angular_velocity"])
        ax[0].set_title('angular velocity')
        ax[0].set_ylabel('radians / second')

        ax[1].plot(self.logs["timevals"], self.logs["linear_velocity"])
        ax[1].set_xlabel('time (seconds)')
        ax[1].set_ylabel('meters / second')
        _ = ax[1].set_title('linear_velocity')
        if save:
            try:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir,"velocities.png"))
            except Exception as e:
                print("Saving failed: ", e)

        fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=dpi)
        ax[0].plot(self.logs["timevals"], np.abs(self.logs["position"]))
        ax[0].set_xlabel('time (seconds)')
        ax[0].set_ylabel('meters')
        _ = ax[0].set_title('position')
        ax[0].set_yscale('log')


        ax[1].plot(np.array(self.logs["position"])[:,0], np.array(self.logs["position"])[:,1])
        ax[1].set_xlabel('meters')
        ax[1].set_ylabel('meters')
        _ = ax[1].set_title('x y coordinates')
        plt.tight_layout()
        if save:
            try:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, "positions.png"))
            except Exception as e:
                print("Saving failed: ", e)

    def saveSimulationData(self, save:bool = True, save_dir:str = "mujoco_experiment", suffix:str="") -> None:
        """
        Saves the simulation data."""

        var_name = ["x","y","z","w"]
        if save:
            try:
                os.makedirs(save_dir, exist_ok=True)
                csv_data = pd.DataFrame()
                for key in self.logs.keys():
                    if len(self.logs[key]) != 0:
                        if key == "actions":
                            data = np.array(self.logs[key])
                            for i in range(data.shape[1]):
                                csv_data["t_"+str(i)] = data[:,i]
                        else:
                            data = np.array(self.logs[key])
                            if len(data.shape) > 1:
                                for i in range(data.shape[1]):
                                    csv_data[var_name[i]+"_"+key] = data[:,i]
                            else:
                                csv_data[key] = data
                csv_data.to_csv(os.path.join(save_dir, "exp_logs"+suffix+".csv"))
                self.csv_datas.append(csv_data)

            except Exception as e:
                print("Saving failed: ", e)

    
    def plotBatch(self, dpi:int = 120, width:int = 600, height:int = 800, save:bool = False, save_dir:str = "mujoco_experiment") -> None:
        figsize = (width / dpi, height / dpi)
        fig = plt.figure(figsize=figsize)
        for csv_data in self.csv_datas:
            plt.plot(csv_data['x_position'], csv_data['y_position'])
        plt.axis("equal")
        plt.xlabel("meters")
        plt.ylabel("meters")
        plt.tight_layout()
        fig.savefig(os.path.join(save_dir, "positions.png"))