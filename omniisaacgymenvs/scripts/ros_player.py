from email.errors import ObsoleteHeaderDefect
from logging import config
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

import os

#from rclpy.node import Node
#from my_msgs.msg import Observation # replace with your observation message type
#from my_msgs.msg import Action # replace with your action message type


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
        from std_msgs.msg import String
        import tf2_ros
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        self.r = rospy.Rate(0.2)
        self.sub = rospy.Subscriber("observation_topic", String, self.callback)
        self.pub = rospy.Publisher("action_topic", ByteMultiArray, queue_size=10)
        self.player = player
    def callback(self, msg):
        print(msg.data)
        trans = self.tfBuffer.lookup_transform(turtle_name, 'turtle1', rospy.Time())
        obs = torch.rand(1, 20, device='cuda')
        #obs = torch.tensor(msg.data)
        action = self.player.get_action(obs, is_deterministic=True)
        action = action.cpu().tolist()
        print(obs, action)
        self.pub.publish("",action)
        self.r.sleep()

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
    rospy.spin()

    env.close()

if __name__ == '__main__':

    # rospy.init_node('my_node')
    # node = MyNode()
    # rospy.spin()
    parse_hydra_configs()
