from typing import Callable, NamedTuple, Optional, Union, List
from collections import deque
import numpy as np
import datetime
import torch
import os

import rospy
from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import PoseStamped, Point

from ros.ros_utills import derive_velocities
from omniisaacgymenvs.mujoco_envs.position_controller_RL import PositionController
from omniisaacgymenvs.mujoco_envs.pose_controller_RL import PoseController
from omniisaacgymenvs.mujoco_envs.linear_velocity_tracker_RL import VelocityTracker, TrajectoryTracker
from omniisaacgymenvs.mujoco_envs.RL_games_model_4_mujoco import RLGamesModel

class RLPlayerNode:
    def __init__(self, model: RLGamesModel, task_id, exp_settings, map:List[int]=[2,5,4,7,6,1,0,3]) -> None:
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

        # Initialize Subscriber and Publisher
        self.pose_sub = rospy.Subscriber("/vrpn_client_node/FP_exp_RL/pose", PoseStamped, self.pose_callback)
        self.goal_sub = rospy.Subscriber("/spacer_floating_platform/goal", Point, self.goal_callback)
        self.action_pub = rospy.Publisher("/spacer_floating_platform/valves/input", ByteMultiArray, queue_size=1)

        # Initialize ROS message for thrusters
        self.my_msg = ByteMultiArray()

        self.end_experiment_after_n_steps = rospy.get_param("end_experiment_at_step", 300)
        self.play_rate = rospy.get_param("play_rate", 5.0)
        self.obs_type = type(self.player.observation_space.sample())

        rospy.on_shutdown(self.shutdown)

    def build_controller(self) -> Union[PositionController, PoseController, VelocityTracker]:
        if self.task_id == 0:
            return PositionController(self.model, self.settings.goal_x, self.settings.goal_y, self.settings.distance_threshold)
        elif self.task_id == 1:
            return PoseController(self.model, self.settings.goal_x, self.settings.goal_y, self.settings.goal_theta, self.settings.distance_threshold)
        elif self.task_id == 2:
            return VelocityTracker()
        elif self.task_id == 3:
            raise NotImplementedError

    def reset(self) -> None:
        """
        Resets the goal and the buffers."""

        self.ready = False
        self.controller = self.build_controller()
        self.get_default_goal()
        self.instantiate_buffers()

    def instantiate_buffers(self) -> None:
        """
        Instantiates the buffers used to store the state variables."""

        # Torch buffers
        self.root_pos = torch.zeros((1,2), dtype=torch.float32, device='cuda')
        self.heading = torch.zeros((1,2), dtype=torch.float32, device='cuda')
        self.lin_vel = torch.zeros((1,2), dtype=torch.float32, device='cuda')
        self.task_data = torch.zeros((1,4), dtype=torch.float32, device='cuda')
        self.ang_vel = torch.zeros((1,2), dtype=torch.float32, device='cuda')
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

        self.my_msg.data = [0,0,0,0,0,0,0,0,0]
        self.pub.publish(self.my_msg)

    def remap_actions(self, actions: torch.Tensor) -> list:
        """
        Remaps the actions from the RL algorithm to the thrusters of the platform."""

        return [actions[i] for i in self.map]

    def pose_callback(self, msg):
        current_time = self.rospy.Time.now()
        # Add current pose and time to the buffer
        self.pose_buffer.append(msg)
        self.time_buffer.append(current_time)

        # Calculate velocities if buffer is filled
        if (len(self.pose_buffer) == self.buffer_size):
            self.get_state_from_optitrack()
            self.ready = True

    def get_state_from_optitrack(self, msg):
        """
        Converts a ROS message to an observation."""

        x_pos = msg.pose.position.x
        y_pos = msg.pose.position.y
        self.root_pos [0,0] = x_pos
        self.root_pos [1,0] = y_pos
        quat = msg.pose.orientation 
        ############# Quaternions convention #############
        #     Isaac Sim Core (QW, QX, QY, QZ)
        #   vrpn_client_node (QX, QY, QZ, QW)
        ##################################################
        q = [quat.w, quat.x, quat.y, quat.z]
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        self.heading[0,0] = cosy_cosp
        self.heading[0,1] = siny_cosp
        linear_vel, angular_vel = derive_velocities(self.time_buffer, self.pose_buffer)
        self.state = {"position": self.root_pos, "orientation": self.heading, "linear_velocity": linear_vel[0,:2], "angular_velocity": angular_vel[0,:2]}
    
    def goal_callback(self, msg):
        """
        Callback for the goal topic. It updates the task data with the new goal data."""
        self.goal_data = msg

    def update_task_data(self):
        """
        Updates the task data based on the task id."""

        if self.task_id == 0: # GoToXY
            self.controller.makeObservationBuffer(self.state)
        elif self.task_id == 1: # GoToPose
            self.controller.makeObservationBuffer(self.state)
        elif self.task_id == 2: # TrackXYVelocity
            self.controller.makeObservationBuffer(self.state)
        elif self.task_id == 3: # TrackXYOVelocity
            raise NotImplementedError
    
    def set_default_goal(self):
        """
        Sets the default goal data."""

        self.goal_data = Point()
        if self.task_id == 0: # GoToXY
            self.controller.setGoal(self.state)
        elif self.task_id == 1: # GoToPose
            self.controller.setGoal(self.state)
        elif self.task_id == 2: # TrackXYVelocity
            self.controller.setGoal(self.state)
        elif self.task_id == 3: # TrackXYOVelocity
            raise NotImplementedError

    def get_action(self, lifting_active = 1):
        self.action = self.controller.getAction(self.state, is_deterministic=True)
        self.action = self.action.cpu().tolist()
        action = self.remap_actions(self.action)
        lifting_active = 1
        action.insert(0, lifting_active)
        self.my_msg.data = action
        self.pub.publish(self.my_msg)

    def print_logs(self):
        print("=========================================")
        print(f"step number: {self.count}")
        print(f"task id: {self.settings.ask_id}")
        print(f"goal: {self.goal_data}")
        print(f"observation: {self.obs_dict['state']}")
        print(f"state: {self.state}")
        print(f"action: {self.action}")

    def update_loggers(self):
        self.obs_buffer.append(self.obs)
        self.act_buffer.append(self.action)

    def save_logs(self):
        save_dir = self.save_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "obs.npy"), np.array(self.obs_buffer))
        np.save(os.path.join(save_dir, "act.npy"), np.array(self.act_buffer))
        np.save(os.path.join(save_dir, "sim_obs.npy"), np.array(self.sim_obs_buffer))

    def run(self): 
        self.rate = rospy.Rate(self.play_rate)

        while (not rospy.is_shutdown()) and (self.count < self.end_experiment_after_n_steps):
            if self.ready:
                self.get_action()
                self.update_loggers()
                self.count += 1
                if self.debug:
                    self.print_logs()
            self.rate.sleep()

        # Saves the logs
        self.save_logs()
        # Kills the thrusters once done
        self.action = [0,0,0,0,0,0,0,0]
        self.get_action(lifting_active=0)
    