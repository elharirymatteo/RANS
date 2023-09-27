
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omegaconf import DictConfig, OmegaConf
import hydra
import os

from omniisaacgymenvs.mujoco_envs.controllers.discrete_LQR_controller import DiscreteController, parseControllerConfig
from omniisaacgymenvs.mujoco_envs.controllers.RL_games_model_4_mujoco import RLGamesModel
from omniisaacgymenvs.mujoco_envs.environments.mujoco_base_env import MuJoCoFloatingPlatform, parseEnvironmentConfig
from omniisaacgymenvs.mujoco_envs.controllers.hl_controllers import hlControllerFactory

@hydra.main(config_name="config_mujoco", config_path="../cfg")
def run(cfg: DictConfig):
    #print_dict(cfg)
    cfg_dict = omegaconf_to_dict(cfg)

    # Create the environment
    env = MuJoCoFloatingPlatform(**parseEnvironmentConfig(cfg_dict))
    
    # Get the low-level controller
    if cfg_dict["use_rl"]:
        assert os.path.exists(cfg_dict["checkpoint"]), "A correct path to a neural network must be provided to infer an RL agent."
        ll_controller = RLGamesModel(config=cfg_dict["train"], model_path=cfg_dict["checkpoint"])
    else:
        ll_controller = DiscreteController(**parseControllerConfig(cfg_dict, env))

    # Get the high-level controller
    hl_controller = hlControllerFactory(cfg_dict, ll_controller)
    
    logs = env.runLoop(hl_controller)

    env.saveSimulationData()
    env.plotSimulation()

if __name__ == '__main__':
    run()