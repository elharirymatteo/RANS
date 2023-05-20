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

from pdb import set_trace as bp
import os

#from rclpy.node import Node
#from my_msgs.msg import Observation # replace with your observation message type
#from my_msgs.msg import Action # replace with your action message type
def quaternion_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def get_observation_from_realsense(msg):
    """
    Convert a ROS message to an observation.
    """
    target_pos = [0., 0., 0.]
    x_pos = msg.pose.position.x
    y_pos = msg.pose.position.y
    z_pos = msg.pose.position.z
    dist_x = x_pos - target_pos[0]
    dist_y = y_pos - target_pos[1]
    dist_z = z_pos - target_pos[2]
    pos_dist = [dist_x, dist_y, dist_z]
    quat = msg.pose.orientation # getting quaternion
    ############# Quaternions convention #############
    #     Isaac Sim Core (QW, QX, QY, QZ)
    #   vrpn_client_node (QX, QY, QZ, QW)
    ##################################################
   # swapping w with z while creating quaternion array from Quaternion object

    q = [quat.w, quat.x, quat.y, quat.z]
    # rot_x =  quat_axis(q, 0) #np.random.rand(3)
    # rot_y =  quat_axis(q, 1)
    # rot_z =  quat_axis(q, 2)
    rot_mat = quaternion_rotation_matrix(q)
    lin_vel = [0., 0., 0.]
    ang_vel = [0., 0., 0.]
    # Cast quaternion to Yaw
    siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
    cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
    # orient_z = torch.arctan2(siny_cosp, cosy_cosp)

    obs = torch.tensor(np.array([dist_x, dist_y, dist_z, 
                        rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], 
                        rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], 
                        rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], 
                        0, 0, 0, 0, 0, 0, cosy_cosp, siny_cosp]), dtype=torch.float32, device='cuda')
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
    def __init__(self, player):
        import rospy
        from std_msgs.msg import ByteMultiArray
        from geometry_msgs.msg import PoseStamped
        import tf2_ros
        self.rospy = rospy
        self.buffer = []
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.sub = rospy.Subscriber("/vrpn_client_node/FPA/pose", PoseStamped, self.callback)
        self.pub = rospy.Publisher("/spacer_floating_platform_a/valves/input", ByteMultiArray, queue_size=10)
        self.player = player
        self.my_msg = ByteMultiArray()
        self.count = 0
        #bp()
        print("Node initialized")

    def callback(self, msg):

        #print(msg)
        #bp()
        print(f'count: {self.count}')

        obs = get_observation_from_realsense(msg)
        #print(obs)
        #trans = self.tfBuffer.lookup_transform(turtle_name, 'turtle1', rospy.Time())
        #obs = torch.rand(1, 20, device='cuda')
        #obs = torch.tensor(msg.data)
        action = self.player.get_action(obs, is_deterministic=True)
        action = action.cpu().tolist()        
        # add lifting action
        lifting_active = 1
        action.insert(0, lifting_active)
        self.my_msg.data = action
        #print(obs, action)
        self.pub.publish(self.my_msg)
        time.sleep(1)
        self.count += 1
        if self.count == 20:
            self.my_msg.data = [0,0,0,0,0,0,0,0,0]
            self.pub.publish(self.my_msg)

            self.rospy.signal_shutdown("Done")
            print("Shutting down node")

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    cfg.checkpoint = "./runs/MFP2DGoToPose/nn/MFP2DGoToPose.pth"
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
    rlg_trainer = RLGTrainer(cfg, cfg_dict)
    rlg_trainer.launch_rlg_hydra(env)
    # _____Create players (model)_____
    player = PpoPlayerDiscrete(cfg_dict['train']['params'])
    player.restore(cfg.checkpoint)
    
    enable_ros_extension()
    import rospy
    
    # _____Create ROS node_____
    rospy.init_node('my_node')
    node = MyNode(player)
    rate = rospy.Rate(1.0)
    count = 0
    
    rospy.spin()

    env.close()

if __name__ == '__main__':

    parse_hydra_configs()
