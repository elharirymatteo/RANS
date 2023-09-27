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

        # Build model using the configuration files
        if config is None:
            self.loadConfig(config_path)
        else:
            self.cfg = config
        self.buildModel()
        self.restore(model_path)

        # Default target and task values
        self.mode = 0
        self.position_target = [0,0,0]
        self.orientation_target = [1,0,0,0]
        self.linear_velocity_target = [0,0,0]
        self.angular_velocity_target = [0,0,0]
        
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
        # Infer task flag from the provided targets
        if target_position is None:
            if target_linear_velocity is None:
                raise ValueError("Cannot make sense of the goal passed to the agent.")
            else:
                if target_angular_velocity is None:
                    self.linear_velocity_target = target_linear_velocity
                    if target_heading is None:
                        self.mode = 2
                    else:
                        self.mode = 4
                        self.orientation_target = target_heading
                else:
                    self.mode = 3
                    self.angular_velocity_target = target_angular_velocity
        else:
            self.position_target = target_position
            if target_heading is None:
                self.mode = 0
            else:
                self.mode = 1
                self.orientation_target = target_heading

    def generate_task_data(self, state):
        if self.mode == 0:
            self.target = [self.position_target[0]-state["position"][0],
                           self.position_target[1]-state["position"][1],
                           0, 0]
        elif self.mode == 1:
            siny_cosp_target = 2 * (self.orientation_target[0] * self.orientation_target[3] + self.orientation_target[1] * self.orientation_target[2])
            cosy_cosp_target = 1 - 2 * (self.orientation_target[2] * self.orientation_target[2] + self.orientation_target[3] * self.orientation_target[3])
            heading_target = np.arctan2(siny_cosp_target, cosy_cosp_target)
            siny_cosp_system = 2 * (state["quaternion"][0] * state["quaternion"][3] + state["quaternion"][1] * state["quaternion"][2])
            cosy_cosp_system = 1 - 2 * (state["quaternion"][2] * state["quaternion"][2] + state["quaternion"][3] * state["quaternion"][3])
            heading_system = np.arctan2(siny_cosp_system, cosy_cosp_system)
            heading_error = np.arctan2(np.sin(heading_target - heading_system), np.cos(heading_target - heading_system))
            self.target = [self.position_target[0]-state["position"][0],
                           self.position_target[1]-state["position"][1],
                           np.cos(heading_error),
                           np.sin(heading_error)]
        elif self.mode == 2:
            self.target = [self.linear_velocity_target[0] - state["linear_velocity"][0],
                           self.linear_velocity_target[1] - state["linear_velocity"][1],
                           0, 0]
        elif self.mode == 3:
            self.target = [self.linear_velocity_target[0] - state["linear_velocity"][0],
                           self.linear_velocity_target[1] - state["linear_velocity"][1],
                           self.angular_velocity_target[2] - state["angular_velocity"][2],
                           0]
        elif self.mode == 4:
            siny_cosp_target = 2 * (self.orientation_target[0] * self.orientation_target[3] + self.orientation_target[1] * self.orientation_target[2])
            cosy_cosp_target = 1 - 2 * (self.orientation_target[2] * self.orientation_target[2] + self.orientation_target[3] * self.orientation_target[3])
            heading_target = np.arctan2(siny_cosp_target, cosy_cosp_target)
            siny_cosp_system = 2 * (state["quaternion"][0] * state["quaternion"][3] + state["quaternion"][1] * state["quaternion"][2])
            cosy_cosp_system = 1 - 2 * (state["quaternion"][2] * state["quaternion"][2] + state["quaternion"][3] * state["quaternion"][3])
            heading_system = np.arctan2(siny_cosp_system, cosy_cosp_system)
            heading_error = np.arctan2(np.sin(heading_target - heading_system), np.cos(heading_target - heading_system))
            self.target = [self.linear_velocity_target[0] - state["linear_velocity"][0],
                           self.linear_velocity_target[1] - state["linear_velocity"][1],
                           np.cos(heading_error),
                           np.sin(heading_error)]
        
    def makeObservationBuffer(self, state):
        self.generate_task_data(state)
        siny_cosp = 2 * (state["quaternion"][0] * state["quaternion"][3] + state["quaternion"][1] * state["quaternion"][2])
        cosy_cosp = 1 - 2 * (state["quaternion"][2] * state["quaternion"][2] + state["quaternion"][3] * state["quaternion"][3]) 
        self.obs_state[0,:2] = torch.tensor([cosy_cosp, siny_cosp], dtype=torch.float32, device="cuda")
        self.obs_state[0,2:4] = torch.tensor(state["linear_velocity"][:2], dtype=torch.float32, device="cuda")
        self.obs_state[0,4] = state["angular_velocity"][2]
        self.obs_state[0,5] = self.mode
        self.obs_state[0,6:] = torch.tensor(self.target, dtype=torch.float32, device="cuda")

    def getAction(self, state, is_deterministic=True, **kwargs):
        self.makeObservationBuffer(state)
        self.obs['state'] = self.obs_state
        actions = self.player.get_action(self.obs.copy(), is_deterministic=is_deterministic).cpu().numpy()
        return actions