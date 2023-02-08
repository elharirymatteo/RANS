import isaacgym
import isaacgymenvs
import torch


from omniisaacgymenvs.utils.hydra_cfg.hydra_utils import *
from omniisaacgymenvs.utils.hydra_cfg.reformat import omegaconf_to_dict, print_dict
from omniisaacgymenvs.utils.rlgames.rlgames_utils import RLGPUAlgoObserver, RLGPUEnv
from omniisaacgymenvs.utils.task_util import initialize_task

envs = isaacgymenvs.make(
	seed=0, 
	task="Ant", 
	num_envs=2000, 
	sim_device="cuda:0",
	rl_device="cuda:0",
)
print("Observation space is", envs.observation_space)
print("Action space is", envs.action_space)
obs = envs.reset()
for _ in range(20):
	obs, reward, done, info = envs.step(
		torch.rand((2000,)+envs.action_space.shape, device="cuda:0")
	)