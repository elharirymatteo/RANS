from collections import deque
import numpy as np
import datetime
import torch
import os

from std_msgs.msg import ByteMultiArray
from geometry_msgs.msg import PoseStamped, Point
import rospy

from ros.ros_utills import derive_velocities

class RLPlayerNode:
    def __init__(self, player, task_id=0, map=[2,5,4,7,6,1,0,3], save_trajectory=True):
        # Initialize variables
        self.buffer_size = 20  # Number of samples for differentiation
        self.pose_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.task_id = task_id

        self.map = map
        self.save_trajectory = save_trajectory
        self.player = player

        self.reset()

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

    def reset(self) -> None:
        """
        Resets the goal and the buffers."""

        self.ready = False
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
        self.obs = torch.zeros((1,10), dtype=torch.float32, device='cuda')
        # Obs dict
        self.obs_dict = dict({'state': self.obs, 'transforms': torch.zeros(5*8, device='cuda'), 'masks': torch.zeros(8, dtype=torch.float32, device='cuda')})
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
        self.act_every += 1

        # Calculate velocities if buffer is filled
        if (len(self.pose_buffer) == self.buffer_size) and (self.act_every == self.buffer_size):
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
            target_position = torch.Tensor([[self.goal_data.point.x, self.goal_data.point.y]], dtype=torch.float32, device="cuda")
            position_error = target_position - self.state["position"]
            self.task_data[:,:2] = position_error
        elif self.task_id == 1: # GoToPose
            target_position = torch.Tensor([[self.goal_data.point.x, self.goal_data.point.y]])
            target_heading = torch.Tensor([[self.goal_data.point.z]], dtype=torch.float32, device="cuda")
            self._position_error = target_position - self.state["position"]
            heading = torch.arctan2(self.state["orientation"][:,1], self.state["orientation"][:, 0])
            heading_error = torch.arctan2(torch.sin(target_heading - heading), torch.cos(target_heading - heading))
            self.task_data[:,:2] = self._position_error
            self.task_data[:, 2] = torch.cos(heading_error)
            self.task_data[:, 3] = torch.sin(heading_error)
        elif self.task_id == 2: # TrackXYVelocity
            target_velocity = torch.Tensor([[self.goal_data.point.x, self.goal_data.point.y]], dtype=torch.float32, device="cuda")
            velocity_error = target_velocity - self.state["linear_velocity"]
            self.task_data[:,:2] = velocity_error
        elif self.task_id == 3: # TrackXYOVelocity
            target_linear_velocity = torch.Tensor([[self.goal_data.point.x, self.goal_data.point.y]], dtype=torch.float32, device="cuda")
            target_angular_velocity = torch.Tensor([[self.goal_data.point.z]], dtype=torch.float32, device="cuda")
            linear_velocity_error = target_linear_velocity - self.state["linear_velocity"]
            angular_velocity_error = target_angular_velocity - self.state["angular_velocity"]
            self.task_data[:,:2] = linear_velocity_error
            self.task_data[:,2] = angular_velocity_error
    
    def get_default_goal(self):
        """
        Sets the default goal data."""

        self.goal_data = Point()
    
    def generate_obs(self):
        """
        Updates the observation tensor with the current state of the robot."""
        self.update_task_data()
        self.obs[:, 0:2] = self.state["orientation"]
        self.obs[:, 2:4] = self.state["linear_velocity"]
        self.obs[:, 4] = self.state["angular_velocity"]
        self.obs[:, 5] = self.task_id
        self.obs[:, 6:10] = self.task_data
        self.obs_dict["state"] = self.obs#dict({'state': self.obs, 'transforms': torch.zeros(5*8, device='cuda'), 'masks': torch.zeros(8, dtype=torch.float32, device='cuda')})

    def send_action(self):
        self.action = self.player.get_action(self.obs, is_deterministic=True)
        self.action = self.action.cpu().tolist()
        action = self.remap_actions(self.action)
        lifting_active = 1
        action.insert(0, lifting_active)
        self.my_msg.data = action
        self.pub.publish(self.my_msg)

    def print(self):
        print("=========================================")
        print(f"step number: {self.count}")
        print(f"task id: {self.task_id}")
        print(f"goal: {self.goal_data}")
        print(f"observation: {self.obs_dict['state']}")
        print(f"state: {self.state}")
        print(f"action: {self.action}")

    def update_loggers(self):
        self.obs_buffer.append(self.obs)
        self.act_buffer.append(self.action)

    def save_logs(self):
        save_dir = "./lab_tests/icra24_Pose/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "obs.npy"), np.array(self.obs_buffer))
        np.save(os.path.join(save_dir, "act.npy"), np.array(self.act_buffer))
        np.save(os.path.join(save_dir, "sim_obs.npy"), np.array(self.sim_obs_buffer))


    def run(self): 
        self.rate = rospy.Rate(self.play_rate)

        while (not rospy.is_shutdown()) and (self.count < self.end_experiment_after_n_steps):
            if self.ready:
                self.generate_obs()
                self.send_action()
                self.update_loggers()
                self.count += 1
                if self.debug:
                    self.print()
            self.rate.sleep()

        # Saves the logs
        self.save_logs()
        # Kills the thrusters once done
        self.my_msg.data = [0,0,0,0,0,0,0,0,0]
        self.pub.publish(self.my_msg)
    