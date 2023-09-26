from typing import Callable, NamedTuple, Optional, Union, List
from collections import deque
import numpy as np
import datetime
import torch
import os

import rospy
from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import PoseStamped, Point, Pose

from ros.ros_utills import derive_velocities
from omniisaacgymenvs.mujoco_envs.legacy.position_controller_RL import PositionController
from omniisaacgymenvs.mujoco_envs.legacy.pose_controller_RL import PoseController
from omniisaacgymenvs.mujoco_envs.legacy.linear_velocity_tracker_RL import VelocityTracker, TrajectoryTracker
from omniisaacgymenvs.mujoco_envs.legacy.pose_controller_DC import PoseController as PoseControllerDC
from omniisaacgymenvs.mujoco_envs.controllers.RL_games_model_4_mujoco import RLGamesModel

class RLPlayerNode:
    def __init__(self, model: RLGamesModel, task_id: int, exp_settings, map:List[int]=[2,5,4,7,6,1,0,3]) -> None:
        """
        Args:
            model (RLGamesModel): The model used for the RL algorithm.
            task_id (int): The id of the task to be performed.
            exp_settings (NamedTuple): The settings of the experiment.
            map (List[int]): The mapping between the thrusters of the platform and the actions of the RL algorithm."""

        # Initialize variables
        self.buffer_size = 30  # Number of samples for differentiation
        self.pose_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.settings = exp_settings

        self.map = map
        self.model = model
        self.task_id = task_id
        self.reset()
        self.controller = self.build_controller()
        self.thruster_mask = np.ones((8), dtype=np.int64)
        for i in self.settings.killed_thruster_id:
            self.thruster_mask[i] = 0

        # Initialize Subscriber and Publisher
        self.pose_sub = rospy.Subscriber("/vrpn_client_node/FP_exp_RL/pose", PoseStamped, self.pose_callback)
        self.goal_sub = rospy.Subscriber("/spacer_floating_platform/goal", Point, self.goal_callback)
        self.action_pub = rospy.Publisher("/spacer_floating_platform/valves/input", ByteMultiArray, queue_size=1)

        # Initialize ROS message for thrusters
        self.my_msg = ByteMultiArray()
        rospy.on_shutdown(self.shutdown)

    def build_controller(self) -> Union[PositionController, PoseController, VelocityTracker, PoseControllerDC]:
        """
        Builds the controller based on the task id.
        
        Returns:
            Union[PositionController, PoseController, VelocityTracker, PoseControllerDC]: The controller."""

        if self.task_id == -1:
            return PoseControllerDC(self.model, self.settings.goal_x, self.settings.goal_y, self.settings.goal_theta, self.settings.distance_threshold)
        elif self.task_id == 0:
            return PositionController(self.model, self.settings.goal_x, self.settings.goal_y, self.settings.distance_threshold)
        elif self.task_id == 1:
            return PoseController(self.model, self.settings.goal_x, self.settings.goal_y, self.settings.goal_theta, self.settings.distance_threshold, self.settings.heading_threshold)
        elif self.task_id == 2:
            tracker = TrajectoryTracker(lookahead=self.settings.lookahead_dist, closed=self.settings.closed, offset=(self.settings.trajectory_x_offset, self.settings.trajectory_y_offset))
            if self.settings.trajectory_type.lower() == "square":
                tracker.generateSquare(h=self.settings.height)
            elif self.settings.trajectory_type.lower() == "circle":
                tracker.generateCircle(radius=self.settings.radius)
            elif self.settings.trajectory_type.lower() == "spiral":
                tracker.generateSpiral(start_radius=self.settings.start_radius, end_radius=self.settings.end_radius, num_loop=self.settings.num_loop)
            else:
                raise ValueError("Unknown trajectory type. Must be square, circle or spiral.")
            return VelocityTracker(tracker, self.model)
        elif self.task_id == 3:
            raise NotImplementedError

    def reset(self) -> None:
        """
        Resets the goal and the buffers."""

        self.ready = False
        self.controller = self.build_controller()
        self.set_default_goal()
        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the state variables."""

        # Torch buffers
        self.root_pos = np.zeros((2), dtype=np.float32)
        self.heading = np.zeros((2), dtype=np.float32)
        self.lin_vel = np.zeros((2), dtype=np.float32)
        self.task_data = np.zeros((4), dtype=np.float32)
        self.ang_vel = np.zeros((2), dtype=np.float32)
        # Obs dict
        self.state = None
        # ROS buffers
        self.count = 0
        self.obs_buffer = []
        self.sim_obs_buffer = []
        self.act_buffer = []

    def shutdown(self) -> None:
        """
        Shutdown the node and kills the thrusters while leaving the air-bearing on."""

        self.my_msg.data = [1,0,0,0,0,0,0,0,0]
        self.action_pub.publish(self.my_msg)
        rospy.sleep(1)
        self.my_msg.data = [0,0,0,0,0,0,0,0,0]
        self.action_pub.publish(self.my_msg)

    def remap_actions(self, actions: torch.Tensor) -> List[float]:
        """
        Remaps the actions from the RL algorithm to the thrusters of the platform.
        
        Args:
            actions (torch.Tensor): The actions from the RL algorithm.
        
        Returns:
            List[float]: The actions for the thrusters."""

        return [actions[i] for i in self.map]

    def pose_callback(self, msg: Pose) -> None:
        """
        Callback for the pose topic. It updates the state of the agent.
        
        Args:
            msg (Pose): The pose message."""
        
        #current_time = rospy.Time.now()
        current_time = msg.header.stamp

        # Add current pose and time to the buffer
        self.pose_buffer.append(msg)
        self.time_buffer.append(current_time)

        # Calculate velocities if buffer is filled
        if (len(self.pose_buffer) == self.buffer_size):
            self.get_state_from_optitrack(msg)
            self.ready = True

    def get_state_from_optitrack(self, msg: Pose) -> None:
        """
        Converts a ROS message to an observation.
        
        Args:
            msg (Pose): The pose message."""

        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        self.root_pos[0] = x_pos
        self.root_pos[1] = y_pos
        quat = msg.pose.orientation 
        ############# Quaternions convention #############
        #     Isaac Sim Core (QW, QX, QY, QZ)
        #   vrpn_client_node (QX, QY, QZ, QW)
        ##################################################
        q = [quat.w, quat.x, quat.y, quat.z]
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        self.heading[0] = cosy_cosp
        self.heading[1] = siny_cosp
        linear_vel, angular_vel = derive_velocities(self.time_buffer, self.pose_buffer)
        if self.task_id == -1:
            self.state = {"position": np.array([x_pos, y_pos, 0]), "quaternion": q, "linear_velocity": [linear_vel[0], linear_vel[1], 0], "angular_velocity": [0, 0, angular_vel[-1]]}
        else:
            self.state = {"position": self.root_pos, "orientation": self.heading, "linear_velocity": linear_vel[:2], "angular_velocity": angular_vel[-1]}
    
    def goal_callback(self, msg: Point) -> None:
        """
        Callback for the goal topic. It updates the task data with the new goal data.
        
        Args:
            msg (Point): The goal message."""
        
        self.goal_data = msg

    def update_task_data(self) -> None:
        """
        Updates the task data based on the task id."""

        if self.task_id == 0: # GoToXY
            self.controller.makeObservationBuffer(self.state)
        elif self.task_id == -1: # GoToPose
            self.controller.makeState4Controller(self.state)
        elif self.task_id == 1: # GoToPose
            self.controller.makeObservationBuffer(self.state)
        elif self.task_id == 2: # TrackXYVelocity
            self.controller.makeObservationBuffer(self.state)
        elif self.task_id == 3: # TrackXYOVelocity
            raise NotImplementedError
    
    def set_default_goal(self) -> None:
        """
        Sets the default goal data."""

        self.goal_data = Point()
        if self.task_id == 0: # GoToXY
            self.controller.setGoal([0,0])
        elif self.task_id == -1: # GoToPose DC
            self.controller.setGoal([0,0,0])
        elif self.task_id == 1: # GoToPose
            self.controller.setGoal([0,0,0])
        elif self.task_id == 2: # TrackXYVelocity
            self.controller.setGoal([0,0])
        elif self.task_id == 3: # TrackXYOVelocity
            raise NotImplementedError

    def get_action(self, lifting_active:int = 1) -> None:
        """
        Gets the action from the RL algorithm and publishes it to the thrusters.
        
        Args:
            lifting_active (int, optional): Whether or not the lifting thruster is active. Defaults to 1."""
        self.action = self.controller.getAction(self.state, is_deterministic=True)
        self.action = self.action * self.thruster_mask
        action = self.remap_actions(self.action)
        lifting_active = 1
        action.insert(0, lifting_active)
        self.my_msg.data = action
        self.action_pub.publish(self.my_msg)

    def print_logs(self) -> None:
        """
        Prints the logs."""

        print("=========================================")
        print(f"step number: {self.count}")
        print(f"task id: {self.task_id}")
        print(f"goal: {self.controller.getGoal()}")
        print(f"observation: {self.controller.getObs()}")
        print(f"state: {self.state}")
        print(f"action: {self.action}")

    def update_loggers(self) -> None:
        """
        Updates the loggers."""

        self.obs_buffer.append(self.controller.getObs())
        self.act_buffer.append(self.action)

    def save_logs(self) -> None:
        """
        Saves the logs."""

        save_dir = self.settings.save_dir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "obs.npy"), np.array(self.obs_buffer))
        np.save(os.path.join(save_dir, "act.npy"), np.array(self.act_buffer))
        np.save(os.path.join(save_dir, "sim_obs.npy"), np.array(self.sim_obs_buffer))

    def update_controller_matrix(self) -> None:
        """
        Updates the controller matrix."""
        if self.update_once:
            r0 = np.concatenate((self.state["position"],self.state["linear_velocity"], self.state["quaternion"],self.state["angular_velocity"]),axis =None)
            print(r0)
            self.controller.model.compute_linearized_system(r0=r0)
            self.update_once = False



    def run(self) -> None:
        """
        Runs the RL algorithm."""
        self.update_once = True
        self.rate = rospy.Rate(self.settings.play_rate)
        start_time = rospy.Time.now()
        run_time = rospy.Time.now() - start_time
        while (not rospy.is_shutdown()) and (run_time.to_sec() < self.settings.exp_duration):
            if self.ready:
                if self.task_id == -1:
                    self.update_controller_matrix()
                self.get_action()
                self.update_loggers()
                self.count += 1
                if self.settings.debug:
                    self.print_logs()
            run_time = rospy.Time.now() - start_time
            self.rate.sleep()
        # Saves the logs
        self.save_logs()
        # Kills the thrusters once done
        self.shutdown()
    