import torch
from dataclasses import dataclass

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

@dataclass
class GoToXYReward:
    reward_mode: str = "linear"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self):
        assert self.reward_mode.lower() in ["linear", "square", "exponential"], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(self, current_state, actions, position_error):
        if self.reward_mode.lower() == "linear":
            position_reward = 1.0 / (1.0 + position_error)
        elif self.reward_mode.lower() == "square":
            position_reward = 1.0 / (1.0 + position_error*position_error)
        elif self.reward_mode.lower() == "exponential":
            position_reward = torch.exp(-position_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return position_reward

@dataclass
class GoToPoseReward:
    position_reward_mode: str = "linear"
    heading_reward_mode: str = "linear"
    position_exponential_reward_coeff: float = 0.25
    heading_exponential_reward_coeff: float = 0.25
    position_scale: float = 1.0
    heading_scale: float = 1.0

    def __post_init__(self):
        assert self.position_reward_mode.lower() in ["linear", "square", "exponential"], "Linear, Square and Exponential are the only currently supported mode."
        assert self.heading_reward_mode.lower() in ["linear", "square", "exponential"], "Linear, Square and Exponential are the only currently supported mode."
    
    def compute_reward(self, current_state, actions, position_error, heading_error):
        if self.position_reward_mode.lower() == "linear":
            position_reward = 1.0 / (1.0 + position_error) * self.position_scale
        elif self.position_reward_mode.lower() == "square":
            position_reward = 1.0 / (1.0 + position_error) * self.position_scale
        elif self.position_reward_mode.lower() == "exponential":
            position_reward = torch.exp( - position_error / 0.25) * self.position_scale
        else:
            raise ValueError("Unknown reward type.")

        if self.heading_reward_mode.lower() == "linear":
            heading_reward = 1.0 / (1.0 + heading_error)  * self.heading_scale
        elif self.heading_reward_mode.lower() == "square":
            heading_reward = 1.0 / (1.0 + heading_error)  * self.heading_scale
        elif self.heading_reward_mode.lower() == "exponential":
            heading_reward = torch.exp( - heading_error / 0.25) * self.heading_scale
        else:
            raise ValueError("Unknown reward type.")
        return position_reward, heading_reward
    
@dataclass
class TrackXYVelocityReward:
    reward_mode: str = "linear"
    exponential_reward_coeff: float = 0.25

    def __post_init__(self):
        assert self.reward_mode.lower() in ["linear", "square", "exponential"], "Linear, Square and Exponential are the only currently supported mode."

    def compute_reward(self, current_state, actions, velocity_error):
        if self.reward_mode.lower() == "linear":
            velocity_reward = 1.0 / (1.0 + velocity_error)
        elif self.reward_mode.lower() == "square":
            velocity_reward = 1.0 / (1.0 + velocity_error*velocity_error)
        elif self.reward_mode.lower() == "exponential":
            velocity_reward = torch.exp(-velocity_error / self.exponential_reward_coeff)
        else:
            raise ValueError("Unknown reward type.")
        return velocity_reward

@dataclass
class TrackXYOVelocityReward:
    linear_reward_mode: str = "linear"
    angular_reward_mode: str = "linear"
    linear_exponential_reward_coeff: float = 0.25
    angular_exponential_reward_coeff: float = 0.25
    linear_scale: float = 1.0
    angular_scale: float = 1.0

    def __post_init__(self):
        assert self.linear_reward_mode.lower() in ["linear", "square", "exponential"], "Linear, Square and Exponential are the only currently supported mode."
        assert self.angular_reward_mode.lower() in ["linear", "square", "exponential"], "Linear, Square and Exponential are the only currently supported mode."
    
    def compute_reward(self, current_state, actions, linear_velocity_error, angular_velocity_error):
        if self.linear_reward_mode.lower() == "linear":
            linear_reward = 1.0 / (1.0 + linear_velocity_error) * self.linear_scale
        elif self.linear_reward_mode.lower() == "square":
            linear_reward = 1.0 / (1.0 + linear_velocity_error) * self.linear_scale
        elif self.linear_reward_mode.lower() == "exponential":
            linear_reward = torch.exp( - linear_velocity_error / 0.25) * self.linear_scale
        else:
            raise ValueError("Unknown reward type.")

        if self.angular_reward_mode.lower() == "linear":
            angular_reward = 1.0 / (1.0 + angular_velocity_error)  * self.angular_scale
        elif self.angular_reward_mode.lower() == "square":
            angular_reward = 1.0 / (1.0 + angular_velocity_error)  * self.angular_scale
        elif self.angular_reward_mode.lower() == "exponential":
            angular_reward = torch.exp( - angular_velocity_error / 0.25) * self.angular_scale
        else:
            raise ValueError("Unknown reward type.")
        return linear_reward, angular_reward