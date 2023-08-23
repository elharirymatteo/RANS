from gym import spaces
import numpy as np
import torch
import yaml

from rl_games.algos_torch.players import BasicPpoPlayerContinuous, BasicPpoPlayerDiscrete


class RLGamesModel:
    def __init__(self, config_path, model_path):
        self.obs = dict({'state':torch.zeros((1,10), dtype=torch.float32, device='cuda'),
                    'transforms': torch.zeros(5,8, device='cuda'),
                    'masks': torch.zeros(8, dtype=torch.float32, device='cuda')})
        self.loadConfig(config_path)
        self.buildModel()
        self.restore(model_path)

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

    def getAction(self, state, is_deterministic=True):
        self.obs['state'] = state
        return self.player.get_action(self.obs, is_deterministic=is_deterministic).cpu().numpy()