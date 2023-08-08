from turtle import heading
import numpy as np
import torch
import hydra
from omegaconf import DictConfig
import datetime
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from rl_games.algos_torch.players import PpoPlayerDiscrete
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from rlgames_train import RLGTrainer
from rl_games.torch_runner import Runner
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames
from omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import torch
import yaml
#from omni.isaac.core.utils.torch.rotations import *
import time
from tasks.utils.fp_utils import quaternion_to_rotation_matrix
from pdb import set_trace as bp
import os
from collections import deque

    
def get_observation_from_realsense(obs_type, task_flag, msg, lin_vel, ang_vel):
    """
    Convert a ROS message to an observation.
    """
    target_pos = [2.5, -1.5, 0.]
    target_heading = 0.
    x_pos = msg.pose.position.x
    y_pos = msg.pose.position.y
    z_pos = msg.pose.position.z
    dist_x = - x_pos + target_pos[0]
    dist_y = - y_pos + target_pos[1]
    dist_z = - z_pos + target_pos[2]
    pos_dist = [dist_x, dist_y, dist_z]
    quat = msg.pose.orientation # getting quaternion
    ############# Quaternions convention #############
    #     Isaac Sim Core (QW, QX, QY, QZ)
    #   vrpn_client_node (QX, QY, QZ, QW)
    ##################################################
  
   # swapping w with z while creating quaternion array from Quaternion object
    q = [quat.w, quat.x, quat.y, quat.z]
    # rot_x =  quat_axis(q, 0) #np.random.rand(3)
    rot_mat = quaternion_to_rotation_matrix(q)

    
    # Cast quaternion to Yaw    
    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])

    if obs_type == np.ndarray: 
        obs = torch.tensor(np.array([dist_x, dist_y, dist_z, 
                            rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], 
                            rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], 
                            rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], 
                            lin_vel[0],lin_vel[1],lin_vel[2],
                            ang_vel[0],ang_vel[1],ang_vel[2], 
                            cosy_cosp, siny_cosp]), dtype=torch.float32, device='cuda')
    
    else:
        # TODO: Add task data based on task_flag, currently only for task 1
        heading = torch.arctan2(siny_cosp, cosy_cosp)
        heading_error = torch.arctan2(torch.sin(target_heading - heading), torch.cos(target_heading - heading)) 
        task_data = [pos_dist[0], pos_dist[1], torch.cos(heading_error), torch.sin(heading_error)]

        obs = dict({'state':torch.tensor([cosy_cosp, siny_cosp, lin_vel[0], lin_vel[1], ang_vel[2], 
                                                   task_flag, task_data[0], task_data[1], task_data[2], task_data[3]], dtype=torch.float32, device='cuda'),
               'transforms': torch.zeros(5*8, device='cuda'), 'masks': torch.zeros(8, dtype=torch.float32, device='cuda')})
        
    return obs 


def enable_ros_extension(env_var: str = "ROS_DISTRO"):
    """
    Enable the ROS extension.
    """

    import omni.ext

    ROS_DISTRO: str = os.environ.get(env_var, "noetic")
    assert ROS_DISTRO in [
        "noetic",
        "foxy",
        "humble",
    ], f"${env_var} must be one of [noetic, foxy, humble]"

    # Get the extension manager and list of available extensions
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extensions = extension_manager.get_extensions()

    # Determine the selected ROS extension id
    if ROS_DISTRO == "noetic":
        ros_extension = [ext for ext in extensions if "ros_bridge" in ext["id"]][0]
    elif ROS_DISTRO in "humble":
        ros_extension = [
            ext
            for ext in extensions
            if "ros2_bridge" in ext["id"] and "humble" in ext["id"]
        ][0]
    elif ROS_DISTRO == "foxy":
        ros_extension = [ext for ext in extensions if "ros2_bridge" in ext["id"]][0]

    # Load the ROS extension if it is not already loaded
    if not extension_manager.is_extension_enabled(ros_extension["id"]):
        extension_manager.set_extension_enabled_immediate(ros_extension["id"], True)

class MyNode:
    def __init__(self, player, task_flag):
        import rospy
        from std_msgs.msg import ByteMultiArray
        from geometry_msgs.msg import PoseStamped
        self.rospy = rospy

        # Initialize variables
        self.buffer_size = 20  # Number of samples for differentiation
        self.pose_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        self.act_every = 0 # act every 5 steps
        self.task_flag = task_flag
        #self.map = [7,0,5,6,3,4,1,2]
        #self.map = [5,6,7,0,1,2,3,4]
        #self.map = [6,5,0,7,2,1,4,3]
        self.map = [2,5,4,7,6,1,0,3]
        #self.map = [5,2,7,4,1,6,3,0]
        #self.map = [3,4,5,6,7,0,1,2]
        #self.map = [4,3,6,5,0,7,2,1]


        # Initialize Subscriber and Publisher
        self.sub = rospy.Subscriber("/vrpn_client_node/FP_exp_RL/pose", PoseStamped, self.callback)
        self.pub = rospy.Publisher("/spacer_floating_platform/valves/input", ByteMultiArray, queue_size=1)

        self.player = player
        self.save_trajectory = True
        self.obs_buffer = []
        self.sim_obs_buffer = []
        self.act_buffer = []
        self.my_msg = ByteMultiArray()
        self.count = 0

        self.end_experiment_at_step = rospy.get_param("end_experiment_at_step", 300)
        self.play_rate = rospy.get_param("play_rate", 5.0)
        self.rate = rospy.Rate(self.play_rate) # 5hz
        self.obs = torch.zeros((1,10), dtype=torch.float32, device='cuda')
        self.ready = False
        self.obs_type = type(self.player.observation_space.sample())

        rospy.on_shutdown(self.shutdown)
        print("Node initialized")


    def shutdown(self):
        self.my_msg.data = [1,0,0,0,0,0,0,0,0]
        self.pub.publish(self.my_msg)

    def remap_actions(self, actions):
        return [actions[i] for i in self.map]

    def callback(self, msg):

        current_time = self.rospy.Time.now()

        # Add current pose and time to the buffer
        self.pose_buffer.append(msg)
        self.time_buffer.append(current_time)
        self.act_every += 1

        # Calculate velocities if buffer is filled
        if (len(self.pose_buffer) == self.buffer_size) and (self.act_every == self.buffer_size):
            lin_vel, ang_vel = self.derive_velocities()
            self.act_every = 0

            self.obs = get_observation_from_realsense(self.obs_type, self.task_flag, msg, lin_vel, ang_vel)
            self.ready = True

    def run(self):
        
        run_once = True
        while (not self.rospy.is_shutdown()) and (self.count < self.end_experiment_at_step):
            #print(f'Im in')
            if self.ready:
                action = self.player.get_action(self.obs, is_deterministic=True)
                action = action.cpu().tolist()
                if self.save_trajectory:
                    self.obs_buffer.append(self.obs)
                    self.act_buffer.append(action)

                action = self.remap_actions(action)
                lifting_active = 1
                action.insert(0, lifting_active)
                self.my_msg.data = action
                self.pub.publish(self.my_msg)
                self.count += 1
                print(f'count: {self.count}')
                print(self.my_msg.data)
                print(f'optitrack obs: {self.obs["state"]}')

                self.ready = False
            self.rate.sleep()
        
        if self.save_trajectory:
            save_dir = "./lab_tests/icra24_Pose/"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "obs.npy"), np.array(self.obs_buffer))
            np.save(os.path.join(save_dir, "act.npy"), np.array(self.act_buffer))
            np.save(os.path.join(save_dir, "sim_obs.npy"), np.array(self.sim_obs_buffer))

        self.my_msg.data = [0,0,0,0,0,0,0,0,0]
        self.pub.publish(self.my_msg)

    def angular_velocities(self, q, dt, N=1):
        q = q[0::N]
        return (2 / dt) * np.array([
            q[:-1,0]*q[1:,1] - q[:-1,1]*q[1:,0] - q[:-1,2]*q[1:,3] + q[:-1,3]*q[1:,2],
            q[:-1,0]*q[1:,2] + q[:-1,1]*q[1:,3] - q[:-1,2]*q[1:,0] - q[:-1,3]*q[1:,1],
            q[:-1,0]*q[1:,3] - q[:-1,1]*q[1:,2] + q[:-1,2]*q[1:,1] - q[:-1,3]*q[1:,0]])

    def derive_velocities(self):
        dt = (self.time_buffer[-1] - self.time_buffer[0]).to_sec() # Time difference between first and last pose
        # Calculate linear velocities
        linear_positions = np.array([[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in self.pose_buffer])
        linear_velocities = np.diff(linear_positions, axis=0) / dt
        average_linear_velocity = np.mean(linear_velocities, axis=0)

        # Calculate angular velocities
        angular_orientations = np.array([[pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z] for pose in self.pose_buffer])
        dt_buff = np.ones((angular_orientations.shape[0] - 1)) * dt / (angular_orientations.shape[0] - 1)
        angular_velocities = self.angular_velocities(angular_orientations, dt_buff)
        #angular_rot_matrices = np.array([quaternion_to_rotation_matrix(orientation) for orientation in angular_orientations])
        #dR_matrices = np.diff(angular_rot_matrices, axis=0) / dt
        #angular_velocities = np.array([(dR[2, 1], dR[0, 2], dR[1, 0]) for dR in dR_matrices])
        average_angular_velocity = np.mean(angular_velocities, axis=1)

        return average_linear_velocity, average_angular_velocity


@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    #cfg.checkpoint = "./runs/MFP2DGoToPose/nn/MFP2DGoToPose.pth"
    
    # set congig params for evaluation
    cfg.task.env.maxEpisodeLength = 300
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # _____Create environment_____
    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)    
    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed
    task = initialize_task(cfg_dict, env)
    # task flag, and integer between 0 and 4.
    #   - 0: GoToXY - 1: GoToPose - 2: TrackXYVelocity - 3: TrackXYOVelocity - 4: TrackXYVelocityMatchHeading
    task_flag = 0 # default to GoToXY
    if "GoToPose" in cfg.checkpoint:
        task_flag = 1
    elif "TrackXYVelocity" in cfg.checkpoint:
        task_flag = 2
    elif "TrackXYOVelocity" in cfg.checkpoint:
        task_flag = 3
    elif "TrackXYVelocityMatchHeading" in cfg.checkpoint:
        task_flag = 4

    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    # _____Create players (model)_____
    player = PpoPlayerDiscrete(cfg_dict['train']['params'])
    player.restore(cfg.checkpoint)
    
    # _____Create ROS node_____
    enable_ros_extension()
    import rospy
    
    rospy.init_node('my_node')
    node = MyNode(player, task_flag)
    node.run()
    #rospy.spin()
    # adding code to collect a ros bag

    env.close()

if __name__ == '__main__':

    parse_hydra_configs()
