__author__ = "Antoine Richard, Matteo El Hariry, Junnosuke Kamohara"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


from omniisaacgymenvs.tasks.MFP.MFP2D_core import (
    Core,
)
from omniisaacgymenvs.tasks.MFP.MFP2D_task_rewards import (
    CloseProximityDockReward,
)
from omniisaacgymenvs.tasks.MFP.MFP2D_task_parameters import (
    CloseProximityDockParameters,
)
from omniisaacgymenvs.tasks.MFP.curriculum_helpers import (
    CurriculumSampler,
)

from omniisaacgymenvs.utils.dock import Dock, DockView
from omni.isaac.core.articulations import ArticulationView
from pxr import Usd

from matplotlib import pyplot as plt
from typing import Tuple
import numpy as np
import wandb
import torch
import math

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)

class CloseProximityDockTask(Core):
    """
    Implements the CloseProximityDock task. The robot has to reach a target position and heading.
    """

    def __init__(
        self,
        task_param: dict,
        reward_param: dict,
        num_envs: int,
        device: str,
    ) -> None:
        super(CloseProximityDockTask, self).__init__(num_envs, device)
        # Task and reward parameters
        self._task_parameters = CloseProximityDockParameters(**task_param)
        self._reward_parameters = CloseProximityDockReward(**reward_param)
        # Curriculum samplers
        self._spawn_dock_mass_sampler = CurriculumSampler(
            self._task_parameters.spawn_dock_mass_curriculum
        )
        self._spawn_dock_space_sampler = CurriculumSampler(
            self._task_parameters.spawn_dock_space_curriculum
        )
        self._spawn_position_sampler = CurriculumSampler(
            self._task_parameters.spawn_position_curriculum
        )
        self.spawn_relative_angle_sampler = CurriculumSampler(
            self._task_parameters.spawn_relative_angle_curriculum
        )
        self._spawn_heading_sampler = CurriculumSampler(
            self._task_parameters.spawn_heading_curriculum
        )
        self._spawn_linear_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_linear_velocity_curriculum
        )
        self._spawn_angular_velocity_sampler = CurriculumSampler(
            self._task_parameters.spawn_angular_velocity_curriculum
        )
        
        # Buffers
        self._goal_reached = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.int32
        )
        self._target_positions = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self._target_orientations = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
        )
        self._target_headings = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self.relative_angle = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        self._task_label = self._task_label * 1

    def create_stats(self, stats: dict) -> dict:
        """
        Creates a dictionary to store the training statistics for the task.
        Args:
            stats (dict): The dictionary to store the statistics.
        """

        torch_zeros = lambda: torch.zeros(
            self._num_envs, dtype=torch.float, device=self._device, requires_grad=False
        )
        if not "position_reward" in stats.keys():
            stats["position_reward"] = torch_zeros()
        if not "position_error" in stats.keys():
            stats["position_error"] = torch_zeros()
        if not "heading_reward" in stats.keys():
            stats["heading_reward"] = torch_zeros()
        if not "heading_error" in stats.keys():
            stats["heading_error"] = torch_zeros()
        if not "boundary_dist" in stats.keys():
            stats["boundary_dist"] = torch_zeros()
        self.log_with_wandb = []
        self.log_with_wandb += self._task_parameters.boundary_penalty.get_stats_name()
        self.log_with_wandb += self._task_parameters.relative_angle_penalty.get_stats_name()
        for name in self._task_parameters.boundary_penalty.get_stats_name():
            if not name in stats.keys():
                stats[name] = torch_zeros()
        for name in self._task_parameters.relative_angle_penalty.get_stats_name():
            if not name in stats.keys():
                stats[name] = torch_zeros()
        return stats

    def get_state_observations(self, current_state: dict) -> torch.Tensor:
        """
        Computes the observation tensor from the current state of the robot.
        Args:
            current_state (dict): The current state of the robot.
        Returns:
            torch.Tensor: The observation tensor.
        """

        # position distance
        self._position_error = self._target_positions - current_state["position"]
        # heading distance
        heading = torch.arctan2(
            current_state["orientation"][:, 1], current_state["orientation"][:, 0]
        )
        self._heading_error = torch.abs(
            torch.arctan2(
                torch.sin(self._target_headings - heading),
                torch.cos(self._target_headings - heading),
            )
        )
        # Encode task data
        self._task_data[:, :2] = self._position_error
        self._task_data[:, 2] = torch.cos(self._heading_error - math.pi)
        self._task_data[:, 3] = torch.sin(self._heading_error - math.pi)
        return self.update_observation_tensor(current_state)
    
    def compute_relative_angle(self, fp_position:torch.Tensor):
        """
        Compute relative angle between FP and anchor point, where is bit behind target location.
        Args:
            fp_position: position of the FP in env coordinate.
        Returns:
            relative_angle: relative angle between FP and anchor point.
        """
        anchor_point = self._target_positions
        anchor_point[:, 0] -= self._task_parameters.goal_to_penalty_anchor_dist * torch.cos(self._target_headings)
        anchor_point[:, 1] -= self._task_parameters.goal_to_penalty_anchor_dist * torch.sin(self._target_headings)
        relative_angle = torch.atan2((fp_position - anchor_point)[:, 1], (fp_position - anchor_point)[:, 0]) - self._target_headings
        relative_angle = torch.atan2(torch.sin(relative_angle), torch.cos(relative_angle)) # normalize angle within (-pi, pi)
        return relative_angle
    
    def compute_relative_angle_mask(self, relative_angle:torch.Tensor):
        """
        Computes the reward reward mask of relative angle.
        If it exceeds boundary_angle, no reward is given.
        Args:
            relative_angle (torch.Tensor): relative angle between FP and anchor point.
        Returns:
            rward mask (torch.Tensor) : reward mask for relative angle.
        """
        if self._reward_parameters.clip_reward:
            return (1 - torch.abs(relative_angle)/self._reward_parameters.boundary_relative_angle) * \
                (torch.abs(relative_angle) <= self._reward_parameters.boundary_relative_angle).to(torch.float32)
        else:
            return torch.ones_like(relative_angle, dtype=torch.float32)

    def compute_reward(
        self, 
        current_state: dict, 
        actions: torch.Tensor, 
        step: int = 0,
    ) -> torch.Tensor:
        """
        Computes the reward for the current state of the robot.
        This method differs from GoToPose task since the task is docking.

        Args:
            current_state (torch.Tensor): The current state of the robot.
            actions (torch.Tensor): The actions taken by the robot.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The reward for the current state of the robot.
        """
            
        # compute reward mask
        self.relative_angle = self.compute_relative_angle(current_state["position"])
        self.reward_mask = self.compute_relative_angle_mask(self.relative_angle)
        
        # position error
        self.position_dist = torch.sqrt(torch.square(self._position_error).sum(-1))
        # heading error (force heading error to be close to pi)
        self.heading_dist = torch.abs(self._heading_error - math.pi)
        
        # boundary penalty
        self.boundary_dist = torch.abs(
            self._task_parameters.kill_dist - self.position_dist
        )
        self.boundary_penalty = self._task_parameters.boundary_penalty.compute_penalty(
            self.boundary_dist, step
        )
        
        # cone shape penalty on fp-dock relative angle
        self.relative_angle_penalty = self._task_parameters.relative_angle_penalty.compute_penalty(self.relative_angle, step)

        # Checks if the goal is reached
        position_goal_is_reached = (
            self.position_dist < self._task_parameters.position_tolerance
        ).int()
        heading_goal_is_reached = (
            self.heading_dist < self._task_parameters.heading_tolerance
        ).int()
        goal_is_reached = position_goal_is_reached * heading_goal_is_reached
        self._goal_reached *= goal_is_reached  # if not set the value to 0
        self._goal_reached += goal_is_reached  # if it is add 1

        # rewards
        (
            self.position_reward,
            self.heading_reward,
        ) = self._reward_parameters.compute_reward(
            current_state, actions, self.position_dist, self.heading_dist
        )
        return self.reward_mask * (self.position_reward + self.heading_reward) - self.boundary_penalty - self.relative_angle_penalty

    def update_kills(self) -> torch.Tensor:
        """
        Updates if the platforms should be killed or not.

        Returns:
            torch.Tensor: Wether the platforms should be killed or not.
        """

        die = torch.zeros_like(self._goal_reached, dtype=torch.long)
        ones = torch.ones_like(self._goal_reached, dtype=torch.long)
        die = torch.where(
            self.position_dist > self._task_parameters.kill_dist, ones, die
        )
        die = torch.where(
            self._goal_reached > self._task_parameters.kill_after_n_steps_in_tolerance,
            ones,
            die,
        )
        return die
    
    def update_collision_termination(self, die:torch.Tensor, contact_state:torch.Tensor):
        """
        Updates if the platforms should be killed or not based on collision status. 

        Args: 
            die(torch.Tensor): Wether the platforms should be killed or not (from update_kills method).
            contact_state (torch.Tensor): norm of contact force.
            
        Returns:
            torch.Tensor: Wether the platforms should be killed or not.
        """
        ones = torch.ones_like(die)
        die = torch.where(contact_state > self._task_parameters.collision_force_tolerance, ones, die)
        return die
    
    def update_relative_angle_termination(self, die:torch.Tensor):
        """
        Updates if the platforms should be killed or not based on relative angle status. 

        Args: 
            die(torch.Tensor): Wether the platforms should be killed or not (from update_kills method).
            
        Returns:
            torch.Tensor: Wether the platforms should be killed or not.
        """
        ones = torch.ones_like(die)
        die = torch.where(torch.abs(self.relative_angle) > self._task_parameters.kill_relative_angle, ones, die)
        return die

    def update_statistics(self, stats: dict) -> dict:
        """
        Updates the training statistics.

        Args:
            stats (dict):The new stastistics to be logged.

        Returns:
            dict: The statistics of the training
        """

        # stats["position_reward"] += self.reward_mask * self.position_reward
        # stats["heading_reward"] += self.reward_mask * self.heading_reward
        stats["position_reward"] += self.position_reward
        stats["heading_reward"] += self.heading_reward
        stats["position_error"] += self.position_dist
        stats["heading_error"] += self.heading_dist
        stats["boundary_dist"] += self.boundary_dist
        stats = self._task_parameters.boundary_penalty.update_statistics(stats)
        stats = self._task_parameters.relative_angle_penalty.update_statistics(stats)
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """
        Resets the goal_reached_flag when an agent manages to solve its task."""

        self._goal_reached[env_ids] = 0

    def get_goals(
        self,
        env_ids: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates a random goal for the task.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The target positions and orientations in env coordinate.
        """
        num_goals = len(env_ids)
        target_positions = torch.zeros(
            (num_goals, 3), device=self._device, dtype=torch.float32
        )
        target_orientations = torch.zeros(
            (num_goals, 4), device=self._device, dtype=torch.float32
        )
        
        # Randomizes the target position (completely random)
        dock_space = self._spawn_dock_space_sampler.sample(num_goals, step, device=self._device) # space between dock face close to wall and wall surface (free space)
        self._target_positions[env_ids, 0] = \
            2*(self._task_parameters.env_x/2 - self._task_parameters.dock_footprint_diameter - dock_space) * torch.rand((num_goals,), device=self._device) \
                - (self._task_parameters.env_x/2 - self._task_parameters.dock_footprint_diameter - dock_space)
        
        self._target_positions[env_ids, 1] = \
            2*(self._task_parameters.env_y/2 - self._task_parameters.dock_footprint_diameter - dock_space) * torch.rand((num_goals,), device=self._device) \
                - (self._task_parameters.env_y/2 - self._task_parameters.dock_footprint_diameter - dock_space)
        
        # Randomizes the target heading
        # First, make dock face the center of environment.
        self._target_headings[env_ids] = torch.atan2(self._target_positions[env_ids, 1], self._target_positions[env_ids, 0]) + math.pi # facing center
        self._target_orientations[env_ids, 0] = torch.cos(
            self._target_headings[env_ids] * 0.5
        )
        self._target_orientations[env_ids, 3] = torch.sin(
            self._target_headings[env_ids] * 0.5
        )

        # Retrieve the target positions and orientations at batch index = env_ids
        target_positions[:, :2] = self._target_positions[env_ids]
        target_positions[:, 2] = torch.ones(num_goals, device=self._device) * 0.45
        target_orientations[:] = self._target_orientations[env_ids]
        
        # Add offset to the local target position
        self._target_positions[env_ids, 0] += (self._task_parameters.fp_footprint_diameter / 2) * torch.cos(self._target_headings[env_ids])
        self._target_positions[env_ids, 1] += (self._task_parameters.fp_footprint_diameter / 2) * torch.sin(self._target_headings[env_ids])

        return target_positions, target_orientations
    
    def set_goals(self, env_ids: torch.Tensor, target_positions: torch.Tensor, target_orientations: torch.Tensor) -> None:
        """
        Update goal attribute of task class.
        Args:
            env_ids: The environment ids for which the goal is set.
            target_positions: The target positions for the robots in env coordinate (world position - env_position).
            target_orientations: The target orientations for the robots."""
        self._target_positions[env_ids] = target_positions[:, :2]
        siny_cosp = 2 * target_orientations[env_ids, 0] * target_orientations[env_ids, 3]
        cosy_cosp = 1 - 2 * (target_orientations[env_ids, 3] * target_orientations[env_ids, 3])
        self._target_headings[env_ids] = torch.arctan2(siny_cosp, cosy_cosp)
        # Add offset to the local target position
        self._target_positions[env_ids, 0] += (self._task_parameters.fp_footprint_diameter / 2) * torch.cos(self._target_headings[env_ids])
        self._target_positions[env_ids, 1] += (self._task_parameters.fp_footprint_diameter / 2) * torch.sin(self._target_headings[env_ids])
    
    def get_initial_conditions(
        self,
        env_ids: torch.Tensor,
        step: int = 0,
    ) -> list:
        """
        Generates spawning positions for the robots following a curriculum.
        [Warmup] Randomize only position, but FP always faces center of FP. 
        [In curriculum] Randomize position and orientation. 
        [After curriculum] Max position and orientation.
        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.reset(env_ids)
        
        # Randomizes the initial position and orientation
        initial_position = torch.zeros(
            (num_resets, 3), device=self._device, dtype=torch.float32
        )
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        relative_angle = self.spawn_relative_angle_sampler.sample(num_resets, step, device=self._device)
        
        initial_position[:, 0] = self._target_positions[env_ids, 0] + r * torch.cos(self._target_headings[env_ids] + relative_angle)
        initial_position[:, 1] = self._target_positions[env_ids, 1] + r * torch.sin(self._target_headings[env_ids] + relative_angle)
        
        initial_orientation = torch.zeros(
            (num_resets, 4), device=self._device, dtype=torch.float32
        )
        heading_noise = self._spawn_heading_sampler.sample(num_resets, step, device=self._device)
        heading_angle = self._target_headings[env_ids] + relative_angle + math.pi + heading_noise
        
        initial_orientation[:, 0] = torch.cos(heading_angle * 0.5)
        initial_orientation[:, 3] = torch.sin(heading_angle * 0.5)
        
        ### Randomize linear and angular velocity ###
        initial_velocity = torch.zeros(
            (num_resets, 6), device=self._device, dtype=torch.float32
        )
        linear_velocity = self._spawn_linear_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        initial_velocity[:, 0] = linear_velocity * torch.cos(theta)
        initial_velocity[:, 1] = linear_velocity * torch.sin(theta)
        
        angular_velocity = self._spawn_angular_velocity_sampler.sample(
            num_resets, step, device=self._device
        )
        initial_velocity[:, 5] = angular_velocity
        
        return (
            initial_position,
            initial_orientation,
            initial_velocity,
        )
    
    def get_dock_masses(self, env_ids: torch.Tensor, step: int = 0) -> torch.Tensor:
        """
        Generates a random mass for the dock.

        Args:
            env_ids (torch.Tensor): The ids of the environments.
            step (int, optional): The current step. Defaults to 0.

        Returns:
            torch.Tensor: The mass of the dock.
        """
        mass = self._spawn_dock_mass_sampler.sample(len(env_ids), step, device=self._device)
        return mass

    def generate_target(self, path, position: torch.Tensor, dock_param: dict = None):
        """
        Generate a docking station where the FP will dock to.
        Args:
            path (str): path to the prim
            position (torch.Tensor): position of the docking station
            dock_param (dict, optional): dictionary of DockParameters. Defaults to None.
        """
        Dock(
            prim_path=path+"/dock", 
            name="dock",
            position=position,
            dock_params=dock_param,
        )
    
    def add_dock_to_scene(
        self, scene: Usd.Stage
        )->Tuple[Usd.Stage, ArticulationView]:
        """
        Adds articulation view and rigiprim view of docking station to the scene.
        Args:
            scene (Usd.Stage): The scene to add the docking station to."""

        dock = DockView(prim_paths_expr="/World/envs/.*/dock")
        scene.add(dock)
        scene.add(dock.base)
        return scene, dock
    
    def log_spawn_data(self, step: int) -> dict:
        """
        Logs the spawn data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The spawn data.
        """

        dict = {}

        num_resets = self._num_envs
        
        r = self._spawn_position_sampler.sample(num_resets, step, device=self._device)
        delta_angle = self.spawn_relative_angle_sampler.sample(num_resets, step, device=self._device)
        heading_noise = self._spawn_heading_sampler.sample(num_resets, step, device=self._device)
        dock_space = self._spawn_dock_space_sampler.sample(num_resets, step, device=self._device)
        dock_mass = self._spawn_dock_mass_sampler.sample(num_resets, step, device=self._device)

        r = r.cpu().numpy()
        delta_angle = delta_angle.cpu().numpy()
        heading_noise = heading_noise.cpu().numpy()
        dock_space = dock_space.cpu().numpy()
        dock_mass = dock_mass.cpu().numpy()
        
        ### Plot spawn mass ###
        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(dock_mass, bins=32)
        ax.set_title("Dock mass")
        ax.set_xlim(
            self._spawn_dock_mass_sampler.get_min_bound(),
            self._spawn_dock_mass_sampler.get_max_bound(),
        )
        ax.set_xlabel("mass (kg)")
        ax.set_ylabel("count")
        fig.tight_layout()
        
        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        dict["curriculum/dock_mass"] = wandb.Image(data)
        
        ### Plot spawn position ###
        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(r, bins=32)
        ax.set_title("Initial position")
        ax.set_xlim(
            self._spawn_position_sampler.get_min_bound(),
            self._spawn_position_sampler.get_max_bound(),
        )
        ax.set_xlabel("spawn distance (m)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        dict["curriculum/initial_position"] = wandb.Image(data)
        
        ### Plot spawn relative heading ###
        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(delta_angle, bins=32)
        ax.set_title("Initial relative heading")
        ax.set_xlim(
            self.spawn_relative_angle_sampler.get_min_bound(),
            self.spawn_relative_angle_sampler.get_max_bound(),
        )
        ax.set_xlabel("angular distance (rad)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_relative_heading"] = wandb.Image(data)
        
        ### Plot spawn heading noise ###
        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(heading_noise, bins=32)
        ax.set_title("Initial heading noise")
        ax.set_xlim(
            self.spawn_relative_angle_sampler.get_min_bound(),
            self.spawn_relative_angle_sampler.get_max_bound(),
        )
        ax.set_xlabel("angular distance (rad)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)

        dict["curriculum/initial_heading_noise"] = wandb.Image(data)
        
        ### Plot dock space ###
        fig, ax = plt.subplots(dpi=100, figsize=(8, 8))
        ax.hist(dock_space, bins=32)
        ax.set_title("Dock space")
        ax.set_xlim(
            self._spawn_dock_space_sampler.get_min_bound(),
            self._spawn_dock_space_sampler.get_max_bound(),
        )
        ax.set_xlabel("spawn distance (m)")
        ax.set_ylabel("count")
        fig.tight_layout()

        fig.canvas.draw()
        data = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        
        dict["curriculum/dock_space"] = wandb.Image(data)
        
        return dict

    def log_target_data(self, step: int) -> dict:
        """
        Logs the target data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The target data.
        """

        return {}

    def get_logs(self, step: int) -> dict:
        """
        Logs the task data to wandb.

        Args:
            step (int): The current step.

        Returns:
            dict: The task data.
        """

        dict = self._task_parameters.boundary_penalty.get_logs()
        if step % 50 == 0:
            dict = {**dict, **self.log_spawn_data(step)}
            dict = {**dict, **self.log_target_data(step)}
        return dict