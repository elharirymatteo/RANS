# write a script that creates a ros node that will run the pretrained .pth model saved in the run folder
# and publish the action to the robot 


import torch
import yaml
from rl_games.torch_runner import Runner

model_path = "./runs/64x64/fp_5kg_1N_5Hz/nn/fp_5kg_1N_5Hz.pth"
config_path = "./runs/64x64/fp_5kg_1N_5Hz/config.yaml"

# Create an instance of the PyTorch model
runner = Runner()
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)

params = config['train']
params['params']['seed'] = 42
print(params)

runner.load(params)
runner.run({
    'train': False,
    'play': True
    })
# Load the saved state dictionary
agent = runner.create_player()
agent.restore(model_path)


# Make sure the loaded object is a PyTorch model object
assert isinstance(agent, torch.nn.Module), "Loaded object is not a PyTorch model"
agent.eval()

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
