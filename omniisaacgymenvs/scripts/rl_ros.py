# write a script that creates a ros node that will run the pretrained .pth model saved in the run folder
# and publish the action to the robot 

import copy
from importlib.util import module_for_loader
import torch
import yaml
from rl_games.torch_runner import Runner
from rl_games.common.a2c_common import A2CBase
from rl_games.common.algo_observer import DefaultAlgoObserver
from rl_games.algos_torch.a2c_continuous import A2CAgent
from rl_games.algos_torch.model_builder import ModelBuilder

model_path = "./runs/64x64/fp_5kg_1N_5Hz/nn/fp_5kg_1N_5Hz.pth"
config_path = "./runs/64x64/fp_5kg_1N_5Hz/config.yaml"

#Create an instance of the PyTorch model
runner = Runner()
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

params = config['train']['params']
params['config']['features'] = {}
params['config']['features']['observer'] = DefaultAlgoObserver()
#params['params']['seed'] = 42

name = config['train']['params']['algo']['name']
mode_builder = ModelBuilder()
model = mode_builder.load(params)


def preproc_obs(obs_batch):
    if type(obs_batch) is dict:
        obs_batch = copy.copy(obs_batch)
        for k,v in obs_batch.items():
            if v.dtype == torch.uint8:
                obs_batch[k] = v.float() / 255.0
            else:
                obs_batch[k] = v
    else:
        if obs_batch.dtype == torch.uint8:
            obs_batch = obs_batch.float() / 255.0
    return obs_batch

obs = {'obs1': torch.randn(18),
        'obs2': torch.randn(18),
        'obs3': torch.randn(18),
        'obs4': torch.randn(18)}
print(f'obs: {obs}')
processed_obs = preproc_obs(obs)

input_dict = {
    'is_train': False,
    'prev_actions': None, 
    'obs' : processed_obs,
    'rnn_states' : None
}
#res_dict = model(input_dict)

model = torch.load('././fuck_nvidia.pt')

print(model)
#print(res_dict)


# runner.load(params)
# # runner.run({
# #     'train': False,
# #     'play': True,
# #     'checkpoint' : model_path,
# #     'sigma' : None
# # })
# # Load the saved state dictionary
# print(runner)

# agent = runner.create_player()
# agent.restore(model_path)


# Make sure the loaded object is a PyTorch model object
#assert isinstance(agent, torch.nn.Module), "Loaded object is not a PyTorch model"
#agent.eval()

#print(model(input))



#!/usr/bin/env python

# import rospy
# import torch
# from my_msgs.msg import Observation # replace with your observation message type
# from my_msgs.msg import Action # replace with your action message type

# class MyNode:
#     def __init__(self):
#         self.model = torch.load("path/to/pretrained/model.pth")
#         self.sub = rospy.Subscriber("observation_topic", Observation, self.callback)
#         self.pub = rospy.Publisher("action_topic", Action, queue_size=10)

#     def callback(self, msg):
#         obs = torch.tensor(msg.data)
#         action = self.model(obs)
#         self.pub.publish(action.tolist())

# if __name__ == '__main__':
#     rospy.init_node('my_node')
#     node = MyNode()
#     rospy.spin()
