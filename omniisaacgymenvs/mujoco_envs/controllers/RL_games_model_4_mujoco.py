from typing import Dict
from gym import spaces
import numpy as np
import torch
import yaml

from rl_games.algos_torch.players import BasicPpoPlayerContinuous, BasicPpoPlayerDiscrete


class RLGamesModel:
    def __init__(self, config: Dict = None,
                       config_path: str = None,
                       model_path: str = None,
                       **kwargs):
        
        self.obs = dict({'state':torch.zeros((1,10), dtype=torch.float32, device='cuda'),
                    'transforms': torch.zeros(5,8, device='cuda'),
                    'masks': torch.zeros(8, dtype=torch.float32, device='cuda')})

        if config is None:
            self.loadConfig(config_path)

        self.buildModel()
        self.restore(model_path)
        self.target = [0,0,0,0]
        self.mode = 0
        self.obs_state = torch.zeros((1,10), dtype=torch.float32, device="cuda")

    def buildModel(self):
        act_space = spaces.Tuple([spaces.Discrete(2)]*8)
        obs_space = spaces.Dict({"state":spaces.Box(np.ones(10) * -np.Inf, np.ones(10) * np.Inf),
                                 "transforms":spaces.Box(low=-1, high=1, shape=(8, 5)),
                                 "masks":spaces.Box(low=0, high=1, shape=(8,))})
        self.player = BasicPpoPlayerDiscrete(self.cfg, obs_space, act_space, clip_actions=False, deterministic=True)

    def loadConfig(self, config_name):
        with open(config_name, 'r') as stream:
            self.cfg = yaml.safe_load(stream)

    def restore(self, model_name):
        self.player.restore(model_name)

    def setTarget(self, target_position=None, target_heading=None, target_linear_velocity=None, target_angular_velocity=None):
        self.mode = 0
        if target_position is None:
            if target_linear_velocity is None:
                raise ValueError("Cannot make sense of the goal passed to the agent.")
            else:
                if target_angular_velocity is None:
                    if target_heading is None:
                        self.mode = 2
                    else:
                        self.mode = 4
                else:
                    self.mode = 3
        else:
            if target_heading is None:
                self.mode = 0
            else:
                self.mode = 1

        siny_cosp = 2 * (target_heading[0] * target_heading[3] + target_heading[1] * target_heading[2])
        cosy_cosp = 1 - 2 * (target_heading[2] * target_heading[2] + target_heading[3] * target_heading[3])

        if self.mode == 0:
            self.target = [target_position[0], target_position[1], 0, 0]
        elif self.mode == 1:
            self.target = [target_position[0], target_position[1], cosy_cosp, siny_cosp]
        elif self.mode == 2:
            self.target = [target_position[0], target_position[1], 0, 0]
        elif self.mode == 3:
            self.target = [target_position[0], target_position[1], target_angular_velocity[2], 0]
        elif self.mode == 4:
            self.target = [target_linear_velocity[0], target_linear_velocity[1], cosy_cosp, siny_cosp]
        
    def makeObservationBuffer(self, state):
        siny_cosp = 2 * (state["quaternion"][0] * state["quaternion"][3] + state["quaternion"][1] * state["quaternion"][2])
        cosy_cosp = 1 - 2 * (state["quaternion"][2] * state["quaternion"][2] + state["quaternion"][3] * state["quaternion"][3]) 
        self.obs_state[0,:2] = torch.tensor([cosy_cosp, siny_cosp], dtype=torch.float32, device="cuda")
        self.obs_state[0,2:4] = torch.tensor(state["linear_velocity"], dtype=torch.float32, device="cuda")
        self.obs_state[0,4] = state["angular_velocity"]
        self.obs_state[0,5] = 2
        self.obs_state[0,6:] = torch.tensor(self.target, dtype=torch.float32, device="cuda")

    def getAction(self, state, is_deterministic=True, **kwargs):
        self.makeObservationBuffer(state)
        self.obs['state'] = state
        return self.player.get_action(self.obs, is_deterministic=is_deterministic).cpu().numpy()