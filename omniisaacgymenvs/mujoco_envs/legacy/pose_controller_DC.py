from typing import Callable, NamedTuple, Optional, Union, List, Dict
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scipy.io
import mujoco
import torch
import os
import cvxpy as cp

from omniisaacgymenvs.mujoco_envs.environments.mujoco_base_env import MuJoCoFloatingPlatform

class MuJoCoPoseControl(MuJoCoFloatingPlatform):
    def __init__(self, step_time:float = 0.02, duration:float = 60.0, inv_play_rate:int = 10,
                 mass:float = 5.32, max_thrust:float = 1.0, radius:float = 0.31) -> None:
        super().__init__(step_time, duration, inv_play_rate, mass, max_thrust, radius)

    def initializeLoggers(self) -> None:
        super().initializeLoggers()
        self.logs["position_target"] = []
        self.logs["heading_target"] = []

    def updateLoggers(self, target) -> None:
        super().updateLoggers()
        self.logs["position_target"].append(target[:2])
        self.logs["heading_target"].append(target[-1])

    def updateState(self) -> Dict[str, np.ndarray]:
        """
        Updates the loggers with the current state of the simulation."""
        state = {}
        state["angular_velocity"] = self.ON.add_noise_on_vel(self.data.qvel[3:6].copy())
        state["linear_velocity"] = self.ON.add_noise_on_vel(self.data.qvel[0:3].copy())
        state["position"] = self.ON.add_noise_on_pos(self.data.qpos[0:3].copy())
        state["quaternion"] = self.data.qpos[3:].copy()
        return state

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
        
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
        ax.plot(self.logs["timevals"], np.array(self.logs["actions"]), label="system action")
        plt.tight_layout()
        if save:
            #try:
                os.makedirs(save_dir, exist_ok=True)
                fig.savefig(os.path.join(save_dir, "actions.png"))
            #except Exception as e:
                #print("Saving failed: ", e)


class DiscreteController:
    """
    Discrete pose controller for the Floating Platform."""

    def __init__(self, target_position: List[float], target_orientation: List[float], thruster_count:int=8, dt:float=0.02, Mod:MuJoCoFloatingPlatform=None, control_type = 'LQR') -> None:
        self.target_position    = np.array(target_position)
        self.target_orientation = np.array(target_orientation)
        self.thruster_count     = thruster_count
        self.thrusters          = np.zeros(thruster_count)  # Initialize all thrusters to off
        self.dt                 = dt

        self.FP                 = Mod
        self.control_type       = control_type
        self.opti_states        = None

        # control parameters
        self.Q = np.diag([1,1,5,5,1,1,1])                      # State cost matrix
        self.R = np.diag([0.01] * self.thruster_count)  # Control cost matrix
        self.W = np.diag([0.1] * 7) # Disturbance weight matrix
        self.find_gains()

    def find_gains(self,r0=None):
        # Compute linearized system matrices A and B based on your system dynamics
        self.A, self.B = self.compute_linearized_system(r0)  # Compute linearized system matrices
        self.make_planar_compatible()

        if self.control_type == 'H-inf':
            self.compute_hinfinity_gains()
        elif self.control_type == 'LQR':
            self.compute_lqr_gains()
        else:
            raise ValueError("Invalid control type specified.")

    def compute_lqr_gains(self):
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.L = np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A

    def compute_hinfinity_gains(self):
        X = cp.Variable((self.A.shape[0], self.A.shape[0]), symmetric=True)
        gamma = cp.Parameter(nonneg=True)  # Define gamma as a parameter

        regularization_param = 1e-6
        # Regularize matrix using the pseudo-inverse
        A_regularized = self.A @ np.linalg.inv(self.A.T @ self.A + regularization_param * np.eye(self.A.shape[1]))
        B_regularized = self.B @ np.linalg.inv(self.B.T @ self.B + regularization_param * np.eye(self.B.shape[1]))

        # Define the constraints using regularized matrices
        constraints = [X >> np.eye(A_regularized.shape[1])]  # X >= 0

        # Define a relaxation factor
        relaxation_factor = 1  # Adjust this value based on your experimentation

        # Linear matrix inequality constraint with relaxation
        constraints += [cp.bmat([[A_regularized.T @ X @ A_regularized - X + self.Q, A_regularized.T @ X @ B_regularized],
                                [B_regularized.T @ X @ A_regularized, B_regularized.T @ X @ B_regularized - (gamma**2) * relaxation_factor * np.eye(B_regularized.shape[1])]]) << 0]

        objective = cp.Minimize(gamma)
        prob = cp.Problem(objective, constraints)

        # Set the value of the parameter gamma
        gamma.value = 1.0  # You can set the initial value based on your problem
        prob.solve()
    
        if prob.status == cp.OPTIMAL:
            self.L = np.linalg.inv(self.B.T @ X.value @ self.B + gamma.value**2 * np.eye(self.B.shape[1])) @ self.B.T @ X.value @ self.A
            breakpoint()
        else:
            raise Exception("H-infinity control design failed.")

    def set_target(self, target_position: List[float], target_orientation: List[float]) -> None:
        """
        Sets the target position and orientation."""

        self.target_position    = np.array(target_position)
        self.target_orientation = np.array(target_orientation)

    def compute_linearized_system(self, r0=None) -> None:
        """
        Compute linearized system matrices A and B.
        With A the state transition matrix.
        With B the control input matrix."""
        if r0 is None:
            r0 = np.concatenate((self.FP.data.qpos[:3],self.FP.data.qvel[:3], self.FP.data.qpos[3:], self.FP.data.qvel[3:]),axis =None) 

              
        t_int   = 0.2 # time-interval at 5Hz
        A       = self.f_STM(r0,t_int,self.FP.model,self.FP.data,self.FP.body_id)
        #Aan     = self.f_STM(r0,t_int,self.FP.model,self.FP.data,self.FP.body_id)
        B       = self.f_B(r0,t_int,self.FP.model,self.FP.data,self.FP.body_id,self.thruster_count)
        return A, B

    def make_planar_compatible(self) -> None:
        """
        Remove elements of the STM to make it planar compatible.
        Required states #[x,y,vx,vy,qw,qz,wz]."""
        
        a = self.A
        b = self.B

        a = np.delete(a, 11, axis=0)  # Remove row: wy
        a = np.delete(a, 10, axis=0)  # Remove row: wx
        a = np.delete(a, 8, axis=0)  # Remove row: qy
        a = np.delete(a, 7, axis=0)  # Remove row: qz
        a = np.delete(a, 5, axis=0)  # Remove row: vz
        a = np.delete(a, 2, axis=0)  # Remove row: z

        a = np.delete(a, 11, axis=1)  # Remove col: wy
        a = np.delete(a, 10, axis=1)  # Remove col: wx
        a = np.delete(a, 8, axis=1)  # Remove col: qy
        a = np.delete(a, 7, axis=1)  # Remove col: qz
        a = np.delete(a, 5, axis=1)  # Remove col: vz
        a = np.delete(a, 2, axis=1)  # Remove col: z

        b = np.delete(b, 11, axis=0)  # Remove row: wy
        b = np.delete(b, 10, axis=0)  # Remove row: wx
        b = np.delete(b, 8, axis=0)  # Remove row: qy
        b = np.delete(b, 7, axis=0)  # Remove row: qz
        b = np.delete(b, 5, axis=0)  # Remove row: vz
        b = np.delete(b, 2, axis=0)  # Remove row: z

        b[b == 0] = 1e-4

        self.A = a
        self.B = b
        return None

    def f_STM(self, r0:np.ndarray, t_int: float, model, data, body_id) -> None:
        """
        Identify A matrix of linearized system through finite differencing."""

        IC_temp0    = r0
        force       = [0.0,0.0,0.0]
        torque      = [0.0,0.0,0.0]

        default_tstep = model.opt.timestep 
        model.opt.timestep = t_int
        current_time = data.time
        for k in range(np.size(r0)):
            delta           = max(1e-3,IC_temp0[k]/100) 
            delta_vec       = np.zeros(np.size(r0))
            delta_vec[k]    = delta
            IC_temp_pos     = np.add(IC_temp0,delta_vec) 
            IC_temp_neg     = np.subtract(IC_temp0,delta_vec) 

            # Positive direction
            data.time               = 0.0
            data.qfrc_applied[...]  = 0.0
            data.qpos[:3]           = IC_temp_pos[0:3]
            data.qvel[:3]           = IC_temp_pos[3:6]           
            data.qpos[3:]           = IC_temp_pos[6:10]
            data.qvel[3:]           = IC_temp_pos[10:13]
            mujoco.mj_applyFT(model, data, force, torque, data.qpos[:3], body_id, data.qfrc_applied)
            mujoco.mj_step(model, data)
            ans_pos         = np.concatenate((data.qpos[:3],data.qvel[:3], data.qpos[3:], data.qvel[3:]),axis =None) 
            #print('final_time', data.time)

            # Negative direction
            data.time               = 0.0
            data.qfrc_applied[...]  = 0.0
            data.qpos[:3]           = IC_temp_neg[0:3]
            data.qvel[:3]           = IC_temp_neg[3:6]           
            data.qpos[3:]           = IC_temp_neg[6:10]
            data.qvel[3:]           = IC_temp_neg[10:13]
            mujoco.mj_applyFT(model, data, force, torque, data.qpos[:3], body_id, data.qfrc_applied)
            mujoco.mj_step(model, data)
            ans_neg         = np.concatenate((data.qpos[:3],data.qvel[:3], data.qpos[3:], data.qvel[3:]),axis =None) 
            #print('final_time', data.time)

            if k==0:
                STM = np.subtract(ans_pos,ans_neg)/(2*delta)
            else :
                temp = np.subtract(ans_pos,ans_neg)/(2*delta)
                STM  = np.vstack((STM,temp))

        STM = STM.transpose()
        STM[6,6] = 1.0

        data.time = current_time
        model.opt.timestep = default_tstep
        return STM

    def f_STM_analytical(self, r0:np.ndarray, t_int:float, model, data, body_id) -> None:
        """        
        Identify A matrix of linearized system through finite differencing."""

        IC_temp0    = r0

        STM = np.eye(np.size(r0))

        w1 = IC_temp0[10]
        w2 = IC_temp0[11]
        w3 = IC_temp0[12]

        qw = IC_temp0[6]
        qx = IC_temp0[7]
        qy = IC_temp0[8]
        qz = IC_temp0[9]

        STM[0,3]  = t_int
        STM[1,4]  = t_int
        STM[2,5]  = t_int

        STM[6,6] = 1 
        STM[6,7] = -0.5*w1*t_int 
        STM[6,8] = -0.5*w2*t_int 
        STM[6,9] = -0.5*w3*t_int 
        STM[6,10] = -0.5*qx*t_int 
        STM[6,11] = -0.5*qy*t_int 
        STM[6,12] = -0.5*qz*t_int 

        STM[7,6] = 0.5*w1*t_int  
        STM[7,7] = 1 
        STM[7,8] = 0.5*w3*t_int 
        STM[7,9] = -0.5*w2*t_int 
        STM[7,10] = 0.5*qw*t_int 
        STM[7,11] = -0.5*qz*t_int 
        STM[7,12] = 0.5*qy*t_int 

        STM[8,6] = 0.5*w2*t_int  
        STM[8,7] = -0.5*w3*t_int  
        STM[8,8] = 1 
        STM[8,9] = 0.5*w1*t_int 
        STM[8,10] = 0.5*qz*t_int 
        STM[8,11] = 0.5*qw*t_int 
        STM[8,12] = -0.5*qx*t_int 

        STM[9,6] = 0.5*w3*t_int  
        STM[9,7] = -0.5*w2*t_int  
        STM[9,8] = -0.5*w1*t_int 
        STM[9,9] = 1
        STM[9,10] = -0.5*qy*t_int 
        STM[9,11] = 0.5*qx*t_int 
        STM[9,12] = 0.5*qw*t_int 
        return STM

    def f_B(self, r0: np.ndarray, t_int: float, model, data, body_id, number_thrust: int) -> None:
        """
        Identify B matrix of linearized system through finite differencing."""

        IC_temp0    = r0
        force       = [0.0,0.0,0.0]
        torque      = [0.0,0.0,0.0]
        default_tstep = model.opt.timestep 
        model.opt.timestep = t_int
        
        u = np.zeros(number_thrust)
        current_time = data.time
        #for k in range(np.size(u)):
        for k in range(np.size(u)):
            delta           = 0.01
            delta_vec       = np.zeros(np.size(u))
            delta_vec[k]    = delta

            # Positive direction
            u_plus                  = np.add(u,delta_vec)
            force_plus              = u_plus[k] * self.FP.forces[k]# * np.sqrt(0.5)
            rmat                    = data.xmat[body_id].reshape(3,3) # Rotation matrix.
            p                       = data.xpos[body_id]              # Position of the body.
            force_plus              = np.matmul(rmat, force_plus)               # Rotate the force to the body frame.
            p2                      = np.matmul(rmat, self.FP.positions[k]) + p     # Compute the position of the force.
            
            data.time               = 0.0
            data.qfrc_applied[...]  = 0.0
            data.qpos[:3]           = IC_temp0[0:3]
            data.qvel[:3]           = IC_temp0[3:6]           
            data.qpos[3:]           = IC_temp0[6:10]
            data.qvel[3:]           = IC_temp0[10:13]
            mujoco.mj_applyFT(model, data, force_plus, torque, p2, body_id, data.qfrc_applied) # Apply the force.
            mujoco.mj_step(model, data)
            ans_pos         = np.concatenate((data.qpos[:3],data.qvel[:3], data.qpos[3:], data.qvel[3:]),axis =None) 

            # Negative direction
            u_minus                 = np.subtract(u,delta_vec)
            force_minus             = u_minus[k] * self.FP.forces[k] * np.sqrt(0.5)
            rmat                    = data.xmat[body_id].reshape(3,3) # Rotation matrix.
            p                       = data.xpos[body_id]              # Position of the body.
            force_minus             = np.matmul(rmat, force_minus)               # Rotate the force to the body frame.
            p2                      = np.matmul(rmat, self.FP.positions[k]) + p     # Compute the position of the force.

            data.time               = 0.0
            data.qfrc_applied[...]  = 0.0
            data.qpos[:3]           = IC_temp0[0:3]
            data.qvel[:3]           = IC_temp0[3:6]           
            data.qpos[3:]           = IC_temp0[6:10]
            data.qvel[3:]           = IC_temp0[10:13]
            mujoco.mj_applyFT(model, data, force_minus, torque, p2, body_id, data.qfrc_applied) # Apply the force.
            mujoco.mj_step(model, data)
            ans_neg         = np.concatenate((data.qpos[:3],data.qvel[:3], data.qpos[3:], data.qvel[3:]),axis =None) 

            if k==0:
                B = np.subtract(ans_pos,ans_neg)/(2*delta)
            else :
                temp = np.subtract(ans_pos,ans_neg)/(2*delta)
                B  = np.vstack((B,temp))

        B = B.transpose()
        model.opt.timestep = default_tstep
        data.time = current_time
        return B

    def control_cost(self) -> np.ndarray:
        # Cost function to be minimized for control input optimization
        if self.control_type == 'H-inf':
            control_input = np.array(self.L @ self.state) + self.disturbance
        elif self.control_type == 'LQR':
            self.find_gains(r0=self.opti_states)
            control_input = np.array(self.L @ self.state) 
        else:
            raise ValueError("Invalid control type specified.")
        return control_input

    def update(self, current_position: np.ndarray, current_orientation: np.ndarray, current_velocity: np.ndarray, current_angular_velocity:np.ndarray, disturbance:np.ndarray = None):
        
        # Calculate errors
        position_error      = self.target_position - current_position
        orientation_error   = self.target_orientation - current_orientation
        
        velocity_error      = np.array([0.0, 0.0, 0.0]) - current_velocity
        angvel_error        = np.array([0.0, 0.0, 0.0]) - current_angular_velocity

        self.opti_states   = np.concatenate((current_position, current_velocity, current_orientation, current_angular_velocity), axis=None)

        if disturbance == None:
            disturbance         = np.random.rand(8) * 0.000                              # Example disturbance 
        self.disturbance    = disturbance

        # Combine errors into the state vector 
        self.state = np.array([position_error[0], position_error[1], velocity_error[0], velocity_error[1], orientation_error[0], orientation_error[3], angvel_error[2]])

        # Optimal U
        original_u = self.control_cost()
        # filter to zero values of u that are less than 0.5
        intermediate_u = np.where(np.abs(original_u) < .25, 0.0, original_u)
        if np.max(intermediate_u) == 0.0:
            normalized_array = np.zeros(self.thruster_count)
        else:
            normalized_array = (intermediate_u - np.min(intermediate_u)) / (np.max(intermediate_u) - np.min(intermediate_u))
        
        # ROund the normalized array to the nearest integer biasing the center to 0.25
        final_U = np.round(normalized_array - 0.25).astype(int)
        # Round the normalized array to the nearest integer

        self.thrusters = final_U
        return self.thrusters


class PoseController:
    """
    Controller for the pose of the robot."""

    def __init__(self, model: DiscreteController, goal_x: List[float], goal_y: List[float], goal_theta: List[float], distance_threshold: float = 0.03) -> None:
        # Discrete controller
        self.model = model
        # Creates an array goals
        if goal_theta is None:
            goal_theta = np.zeros_like(goal_x)
        self.goals = np.array([goal_x, goal_y, goal_theta]).T

        self.current_goal = self.goals[0]
        self.current_goal_controller = np.zeros((3), dtype=np.float32)
        self.current_goal_controller[:2] = self.current_goal[:2]

        self.distance_threshold = distance_threshold

        self.obs_state = torch.zeros((1,10), dtype=torch.float32, device="cuda")

    def isGoalReached(self, state: Dict[str, np.ndarray]) -> bool:
        dist = np.linalg.norm(self.current_goal[:2] - state["position"][:2])
        if dist < self.distance_threshold:
            return True
    
    def getGoal(self) -> np.ndarray:
        return self.current_goal
    
    def setGoal(self, goal:np.ndarray) -> None:
        self.current_goal = goal
        self.goals = np.array([goal])

    def isDone(self) -> bool:
        return len(self.goals) == 0
    
    def getObs(self):
        return self.obs_state.cpu().numpy()

    def makeObservationBuffer(self, state):
        q = state["quaternion"]
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        self.obs_state[0,:2] = torch.tensor([cosy_cosp, siny_cosp], dtype=torch.float32, device="cuda")
        self.obs_state[0,2:4] = torch.tensor(state["linear_velocity"][:2], dtype=torch.float32, device="cuda")
        self.obs_state[0,4] = state["angular_velocity"][-1]
        self.obs_state[0,5] = 1
        self.obs_state[0,6:8] = torch.tensor(self.current_goal[:2] - state["position"][:2], dtype=torch.float32, device="cuda")
        heading = np.arctan2(siny_cosp, cosy_cosp)
        heading_error = np.arctan2(np.sin(self.current_goal[-1] - heading), np.cos(self.current_goal[-1] - heading))
        self.obs_state[0,8] = torch.tensor(np.cos(heading_error), dtype=torch.float32, device="cuda")
        self.obs_state[0,9] = torch.tensor(np.sin(heading_error), dtype=torch.float32, device="cuda")


    def makeState4Controller(self, state: Dict[str, np.ndarray]) -> List[np.ndarray]:
        self.makeObservationBuffer(state)
        current_position = state["position"]
        current_position[-1] = 0
        current_orientation = state["quaternion"]
        current_linear_velocity = state["linear_velocity"]
        current_angular_velocity = state["angular_velocity"]
        return current_position, current_orientation, current_linear_velocity, current_angular_velocity

    def getAction(self, state: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        if self.isGoalReached(state):
            print("Goal reached!")
            if len(self.goals) > 1:
                self.current_goal = self.goals[1,:2]
                self.current_goal_controller[:2] = self.current_goal
                self.goals = self.goals[1:]   
                self.model.find_gains(r0=self.opti_states)    
            else:
                self.goals = []
        current_position, current_orientation, current_linear_velocity, current_angular_velocity = self.makeState4Controller(state)
        self.model.set_target(self.current_goal_controller, [1,0,0,0])
        return self.model.update(current_position, current_orientation, current_linear_velocity, current_angular_velocity)


def parseArgs():
    parser = argparse.ArgumentParser("Generates meshes out of Digital Elevation Models (DEMs) or Heightmaps.")
    parser.add_argument("--goal_x", type=float, nargs="+", default=None, help="List of x coordinates for the goals to be reached by the platform.")
    parser.add_argument("--goal_y", type=float, nargs="+", default=None, help="List of y coordinates for the goals to be reached by the platform.")
    parser.add_argument("--goal_theta", type=float, nargs="+", default=None, help="List of headings for the goals to be reached by the platform. In world frame, radiants.")
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
    # Creates the environment
    print(1.0/args.sim_rate)
    env = MuJoCoPoseControl(step_time=1.0/args.sim_rate, duration=args.sim_duration, inv_play_rate=int(args.sim_rate/args.play_rate),
                            mass=args.platform_mass, radius=args.platform_radius, max_thrust=args.platform_max_thrust)
    # Instantiates the Discrete Controller (DC)
    model = DiscreteController([2.5,-1.5,0.],[1,0,0,0], Mod=env, control_type='LQR') # control type: 'H-inf' or 'LQR' | H-inf not stable at many locations
    #  Creates the velocity tracker
    position_controller = PoseController(model, args.goal_x, args.goal_y, args.goal_theta)
    # Runs the simulation
    env.runLoop(position_controller, [0,0])
    # Plots the simulation
    env.plotSimulation(save_dir = args.save_dir)
    # Saves the simulation data
    env.saveSimulationData(save_dir = args.save_dir)
