import argparse
import os

from omniisaacgymenvs.mujoco_envs.RL_games_model_4_mujoco import RLGamesModel
from omniisaacgymenvs.ros.ros_nodes import RLPlayerNode


if __name__ == '__main__':
    # Initialize the model.
    model = RLGamesModel()
    # Initialize the node.
    node = RLPlayerNode(model)
    # Run the node.
    node.run()