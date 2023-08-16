import hydra

from rl_games.algos_torch.players import PpoPlayerDiscrete
from rlgames_train import RLGTrainer
from omegaconf import DictConfig

from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.task_util import initialize_task
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

@hydra.main(config_name="config", config_path="../cfg")
def parse_hydra_configs(cfg: DictConfig):
    
    # set congig params for evaluation
    cfg.task.env.maxEpisodeLength = 300
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # _____Create environment_____
    headless = cfg.headless
    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = VecEnvRLGames(headless=headless, sim_device=cfg.device_id, enable_livestream=cfg.enable_livestream, enable_viewport=enable_viewport)

    from omni.isaac.core.utils.torch.maths import set_seed
    from ros.ros_utills import enable_ros_extension
    from ros_nodes import RLPlayerNode

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

    env.close()

if __name__ == '__main__':
    parse_hydra_configs()
