from typing import Callable, NamedTuple, Optional, Union, List, Dict
from scipy.linalg import solve_discrete_are
import numpy as np
import mujoco
import cvxpy as cp

from omniisaacgymenvs.mujoco_envs.environments.mujoco_base_env import MuJoCoFloatingPlatform


def parseControllerConfig(cfg_dict: Dict, env:MuJoCoFloatingPlatform) -> Dict[str, Union[List[float], int, float, str, MuJoCoFloatingPlatform]]:
    config = {}
    config["target_position"] = [0,0,0]
    config["target_orientation"] = [1,0,0,0]
    config["target_linear_velocity"] = [0,0,0]
    config["target_angular_velocity"] = [0,0,0]
    config["thruster_count"] = cfg_dict["task"]["env"]["platform"]["configuration"]["num_anchors"]*2
    config["dt"] = cfg_dict["task"]["sim"]["dt"]
    config["Mod"] = env
    config["control_type"] = cfg_dict["controller"]["control_type"]
    config["Q"] = cfg_dict["controller"]["Q"]
    config["R"] = cfg_dict["controller"]["R"]
    config["W"] = cfg_dict["controller"]["W"]
    return config

class DiscreteController:
    """
    Discrete pose controller for the Floating Platform."""

    def __init__(self, target_position: List[float] = [0,0,0],
                       target_orientation: List[float] = [1,0,0,0],
                       target_linear_velocity: List[float] = [0,0,0],
                       target_angular_velocity: List[float] = [0,0,0],
                       thruster_count: int = 8,
                       dt: float = 0.02,
                       Mod: MuJoCoFloatingPlatform = None,
                       control_type: str = 'LQR',
                       Q: List[float] = [1,1,5,5,1,1,1],
                       W: List[float] = [0.01,0.01,0.01,0.01,0.01,0.01,0.01],
                       R: List[float] = [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],
                       **kwargs) -> None:

        self.thruster_count = thruster_count
        self.thrusters = np.zeros(thruster_count)  # Initialize all thrusters to off
        self.dt = dt

        self.FP = Mod
        self.control_type = control_type
        self.opti_states = None

        # Instantiate goals to be null
        self.target_position = target_position
        self.target_orientation = target_orientation
        self.target_linear_velocity = target_linear_velocity
        self.target_angular_velocity = target_angular_velocity

        # Control parameters
        # State cost matrix
        self.Q = np.diag(Q)
        # Control cost matrix
        self.R = np.diag(R)
        # Disturbance weight matrix
        self.W = np.diag(W)
        self.findGains()

    def findGains(self,r0=None) -> None:
        # Compute linearized system matrices A and B based on your system dynamics
        self.A, self.B = self.computeLinearizedSystem(r0)  # Compute linearized system matrices
        self.makePlanarCompatible()

        if self.control_type == 'H-inf':
            self.computeHInfinityGains()
        elif self.control_type == 'LQR':
            self.computeLQRGains()
        else:
            raise ValueError("Invalid control type specified.")

    def computeLQRGains(self) -> None:
        self.P = solve_discrete_are(self.A, self.B, self.Q, self.R)
        self.L = np.linalg.inv(self.R + self.B.T @ self.P @ self.B) @ self.B.T @ self.P @ self.A

    def computeHInfinityGains(self) -> None:
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

    def setTarget(self, target_position: List[float] = None,
                        target_heading: List[float] = None,
                        target_linear_velocity: List[float] = None,
                        target_angular_velocity: List[float] = None) -> None:
        """
        Sets the target position, orientation, and velocities."""

        if target_position is not None:
            self.target_position = np.array(target_position)
        if target_heading is not None:
            self.target_orientation = np.array(target_heading)
        if target_linear_velocity is not None:
            self.target_linear_velocity = np.array(target_linear_velocity)
        if target_angular_velocity is not None:
            self.target_angular_velocity = np.array(target_angular_velocity)

    def computeLinearizedSystem(self, r0: np.ndarray = None) -> None:
        """
        Compute linearized system matrices A and B.
        With A the state transition matrix.
        With B the control input matrix."""

        if r0 is None:
            r0 = np.concatenate((self.FP.data.qpos[:3],self.FP.data.qvel[:3], self.FP.data.qpos[3:], self.FP.data.qvel[3:]),axis =None) 

        t_int   = 0.2 # time-interval at 5Hz
        A       = self.f_STM(r0,t_int,self.FP.model,self.FP.data,self.FP.body_id)
        B       = self.f_B(r0,t_int,self.FP.model,self.FP.data,self.FP.body_id,self.thruster_count)
        return A, B

    def makePlanarCompatible(self) -> None:
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

    def controlCost(self) -> np.ndarray:
        # Cost function to be minimized for control input optimization
        if self.control_type == 'H-inf':
            control_input = np.array(self.L @ self.state) + self.disturbance
        elif self.control_type == 'LQR':
            self.findGains(r0=self.opti_states)
            control_input = np.array(self.L @ self.state) 
        else:
            raise ValueError("Invalid control type specified.")
        return control_input
    
    def makeState4Controller(self, state: Dict[str, np.ndarray]) -> List[np.ndarray]:
        current_position = state["position"]
        current_position[-1] = 0
        current_orientation = state["quaternion"]
        current_linear_velocity = state["linear_velocity"]
        current_angular_velocity = state["angular_velocity"]
        return current_position, current_orientation, current_linear_velocity, current_angular_velocity

    def getAction(self, obs_state, is_deterministic=True):
        return self.update(*self.makeState4Controller(obs_state))
                  
    def update(self, current_position: np.ndarray, current_orientation: np.ndarray, current_velocity: np.ndarray, current_angular_velocity:np.ndarray, disturbance:np.ndarray = None): 
        # Calculate errors
        position_error = self.target_position - current_position
        orientation_error = self.target_orientation - current_orientation        
        velocity_error = self.target_linear_velocity - current_velocity
        angvel_error = self.target_angular_velocity - current_angular_velocity

        self.opti_states = np.concatenate((current_position, current_velocity, current_orientation, current_angular_velocity), axis=None)

        if disturbance == None:
            disturbance = np.random.rand(8) * 0.000 
        self.disturbance = disturbance

        # Combine errors into the state vector (planar)
        self.state = np.array([position_error[0], position_error[1], velocity_error[0], velocity_error[1], orientation_error[0], orientation_error[3], angvel_error[2]])

        # Optimal U
        original_u = self.controlCost()
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