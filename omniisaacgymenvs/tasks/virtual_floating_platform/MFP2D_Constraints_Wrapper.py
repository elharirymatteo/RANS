__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.virtual_floating_platform import MFP2DVirtual
import numpy as np


class MFP2DConstrainedWrapper(MFP2DVirtual):
    def __init__(self, MFP2DVirtual, constraints_to_enforce=None, agent_angular_speed_limit=0.5,
                 agent_energy_limit=0.01, constraint_rates_to_add_as_obs=None):
        super().__init__(MFP2DVirtual)

        self.possible_constraints = ['is_above_angular_speed_limit', 'is_above_energy_limit']

        if constraints_to_enforce is not None:
            self.constraints_to_enforce = constraints_to_enforce
        else:
            self.constraints_to_enforce = []
        
        if constraint_rates_to_add_as_obs is None:
            self.constraint_rates_to_add_as_obs = []
        else:
            self.constraint_rates_to_add_as_obs = constraint_rates_to_add_as_obs

        assert agent_angular_speed_limit >= 0. and agent_angular_speed_limit <= 1., "agent_angular_speed_limit must be between 0 and 1"
        self.agent_angular_speed_limit = agent_angular_speed_limit
        assert agent_energy_limit >= 0. and agent_energy_limit <= 1., "agent_energy_limit must be between 0 and 1"
        self.agent_energy_limit = agent_energy_limit

    def _add_constraint_rates_to_obs(self):
        """
        Constraint rates to track to make this behavior easier to learn for the agent.
        """
        self.constraint_violation_counters = {constraint_name: 0 for constraint_name in self.constraints_to_enforce}
        # some code to create the expanded observation space in case of need
        # ...

    def indicator(self, constraint_name, observation, done):
        """
        Defines, based on the observation vector, the indicator function for each constraint to enforce in the environment.
        :param constraint_name: name of the constraint to enforce
        :param observation: observation vector
        :param done: boolean indicating whether the episode is over
        :return: indicator function value
        """
        if constraint_name == 'is_above_angular_speed_limit':
            return float(np.any(observation["ang_vel"] > self.agent_angular_speed_limit)) 
        elif constraint_name == 'is_above_energy_limit':
            return float(observation["energy"] > self.agent_energy_limit)
        else:
            raise NotImplementedError
        
    def step(self, action):
        """
        Performs a step in the environment.
        :param action: action to perform
        :return: observation, reward, done, info
        """
        # take env step
        next_observation, reward, done, info = self.MFP2DVirtual.step(action)

        # add constraint rates to info
        for constraint_name in self.constraints_to_enforce:
            info[constraint_name] = self.indicator(constraint_name, next_observation, done)

        # increase the constraint violation counter if any constraint is violated
        for constraint_name in self.possible_constraints:
            self.constraint_violation_counters[constraint_name] += self.indicator(constraint_name, next_observation, done)

        return next_observation, reward, done, info
    
    def reset(self, **kwargs):
        """
        Resets the environment.
        :return: observation
        """
         # reset env
        observation = self.MFP2DVirtual.reset(**kwargs)
        # reset constraint violation counters
        self.constraint_violation_counters = {constraint_name: 0 for constraint_name in self.constraints_to_enforce}

        return observation