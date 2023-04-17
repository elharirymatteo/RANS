
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.modular_floating_platform import ModularFloatingPlatform
from omniisaacgymenvs.robots.articulations.views.modular_floating_platform_view import ModularFloatingPlatformView
from omniisaacgymenvs.tasks.utils.fp_utils import quantize_tensor_values

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import omni
import time
import math
import torch
from gym import spaces

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class MFP2DTrackXYVelocityMatchHeadingTask(RLTask):
    def __init__(
        self,
        name: str,                # name of the Task
        sim_config,    # SimConfig instance for parsing cfg
        env,          # env instance of VecEnvBase or inherited class
        offset=None               # transform offset in World
    ) -> None:
         
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["discreteActions"]

        # Platform parameters
        self.mass = self._task_cfg["env"]["mass"]
        self.thrust_force = self._task_cfg["env"]["thrustForce"]
        self.dt = self._task_cfg["sim"]["dt"]
        # Task parameters
        self.xy_velocity_tolerance = self._task_cfg["env"]["task_parameters"]["XYVelocityTolerance"]
        self.kill_after_n_steps_in_tolerance = self._task_cfg["env"]["task_parameters"]["KillAfterNStepsInTolerance"]
        self._min_xy_velocity = self._task_cfg["env"]["task_parameters"]["MinVelocity"]
        self._max_xy_velocity = self._task_cfg["env"]["task_parameters"]["MaxVelocity"]
        self._max_distance = self._task_cfg["env"]["task_parameters"]["MaxDistance"]
        self._reset_error = self._task_cfg["env"]["task_parameters"]["ResetError"]
        # Rewards parameters
        self.rew_scales = {}
        self.rew_scales["xy_velocity"] = self._task_cfg["env"]["learn"]["XYVelocityRewardScale"]
        self.use_linear_rewards = self._task_cfg["env"]["learn"]["UseLinearRewards"]
        self.use_square_rewards = self._task_cfg["env"]["learn"]["UseSquareRewards"]
        self.use_exponential_rewards = self._task_cfg["env"]["learn"]["UseExponentialRewards"]

        self._num_observations = 20

        # define action space
        if self._discrete_actions=="MultiDiscrete":    
            self._num_actions = 8
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)]*8)
        elif self._discrete_actions=="Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            self._num_actions = 8

        self._fp_position = torch.tensor([0, 0., 0.5])
        self._ball_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name, env)

        self.thrust_max = torch.tensor(self.thrust_force, device=self._device, dtype=torch.float32)
        self.target_velocities = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_orient = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device, dtype=torch.float32)
        self.goal_reached = torch.zeros((self.num_envs), device=self._device, dtype=torch.int32)
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
     
        # Extra info
        self.extras = {}

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"xy_velocity_reward": torch_zeros(), "xy_velocity_error": torch_zeros()}
        return

    def set_up_scene(self, scene) -> None:
        self.get_floating_platform()

        RLTask.set_up_scene(self, scene) 

        root_path = "/World/envs/.*/Modular_floating_platform" 
        self._platforms = ModularFloatingPlatformView(prim_paths_expr=root_path, name="modular_floating_platform_view") 
        # set fp base masses according to the task config
        masses = torch.tensor(self.mass, device=self._device, dtype=torch.float).repeat(self.num_envs)
        self._platforms.base.set_masses(masses)

        # Add views to scene
        scene.add(self._platforms) # add view to scene for initialization
        scene.add(self._platforms.thrusters)
        
        space_margin = " "*25
        print("\n########################  Floating platform set up ####################### \n")
        print(f'{space_margin} Number of thrusters: {int(self._platforms.thrusters.count/self._num_envs)}')
        print(f'{space_margin} Mass base: {self._platforms.base.get_masses()[0]:.2f} kg')
        masses = self._platforms.thrusters.get_masses()
        for i in range(int(self._platforms.thrusters.count/self._num_envs)):
            print(f'{space_margin} Mass thruster {i+1}: {masses[i]:.2f} kg')
        print(f'{space_margin} Thrust force: {self.thrust_force} N')
        print("\n##########################################################################")
        return

    def get_floating_platform(self):
        fp = ModularFloatingPlatform(prim_path=self.default_zero_env_path + "/Modular_floating_platform", name="modular_floating_platform",
                            translation=self._fp_position)
        
        self._sim_config.apply_articulation_settings("modular_floating_platform", get_prim_at_path(fp.prim_path),
                                                        self._sim_config.parse_actor_config("modular_floating_platform"))

    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        self.root_pos, self.root_rot = self._platforms.get_world_poses(clone=False)
        self.root_velocities = self._platforms.get_velocities(clone=False)
        root_quats = self.root_rot
        # Get velocity delta to the goal
        self.obs_buf[..., 0:3] = self.target_velocities - self.root_velocities[..., 0:3]
        # Get rotation matrix from quaternions
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z
        # Get velocities in the world frame 
        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]
        self.obs_buf[..., 12:15] = root_linvels
        self.obs_buf[..., 15:18] = root_angvels
        # Get the target heading
        self.obs_buf[..., 18:19] = torch.cos(self.target_orient)
        self.obs_buf[..., 19:20] = torch.sin(self.target_orient)

        observations = {
            self._platforms.name: {
               "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps
        # If is not playing skip
        if not self._env._world.is_playing():
            return                
        # Check which environment need to be reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Reset the environments (Robots)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # Reset the targets (Goals)
        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        # Collect actions
        actions = actions.clone().to(self._device)
        self.actions = actions
        # Remap actions to the correct values
        if self._discrete_actions=="MultiDiscrete":
            # If actions are multidiscrete [0, 1]
            thrust_cmds = self.actions.float()
        elif self._discrete_actions=="Continuous":
            # Transform continuous actions to [0, 1] discrete actions.
            thrust_cmds = torch.clamp((self.actions+1)/2, min=0.0, max=1.0)
        else:
            raise NotImplementedError("")
        
        # Applies the thrust multiplier
        thrusts = self.thrust_max * thrust_cmds
        self.thrusts[:,:,2] = thrusts
        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0
        # Apply forces
        self._platforms.thrusters.apply_forces(self.thrusts, is_global=False)
        return

    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._platforms.get_world_poses()
        self.root_velocities = self._platforms.get_velocities()
        self.dof_pos = self._platforms.get_joint_positions()
        self.dof_vel = self._platforms.get_joint_velocities()

        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, self._num_actions, 3), dtype=torch.float32, device=self._device)
        
        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # Randomizes the position of the ball on the x y axis
        self.target_velocities[envs_long, 0:2] = torch_rand_float(-self._min_xy_velocity, self._max_xy_velocity, (num_sets, 2), device=self._device) 
        self.target_orient[envs_long, 0] = torch.rand(num_sets, device=self._device) * math.pi

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.goal_reached[env_ids] = 0
        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._platforms.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0
        # Randomizes the starting position of the platform
        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        # Randomizes the heading of the platform
        root_rot = self.initial_root_rot.clone()
        random_orient = torch.rand(num_resets, device=self._device) * math.pi
        root_rot[env_ids, 0] = torch.cos(random_orient*0.5)
        root_rot[env_ids, 3] = torch.sin(random_orient*0.5)
        # Sets the velocities to 0
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._platforms.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._platforms.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._platforms.set_world_poses(root_pos[env_ids], root_rot[env_ids], indices=env_ids)
        self._platforms.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(
                self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.

    def calculate_metrics(self) -> None:
        # Distance to their origin
        self.root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot 
        self.root_dist = torch.sqrt(torch.square(self.root_positions).mean(-1))
        orient_z = torch.cos(root_quats[:, 0]) * torch.sin(root_quats[:, 1]) * torch.cos(root_quats[:, 2]) + torch.sin(root_quats[:, 0]) * torch.cos(root_quats[:, 1]) * torch.sin(root_quats[:, 2])
        # linear velocity error
        self.target_dist = torch.sqrt(torch.square(self.target_velocities[:,:2] - self.root_velocities[:,:2]).mean(-1))

        # orientation error
        self.orient_z = orient_z
        self.orient_dist = torch.abs(torch.arctan2(torch.sin(self.target_orient[:,0] - orient_z), torch.cos(self.target_orient[:,0] - orient_z)))

        # Checks if the goal is reached
        goal_is_reached = (self.target_dist < self.xy_velocity_tolerance).int()
        goal_is_reached *= (self.orient_dist < self.heading_tolerance).int()
        self.goal_reached *= goal_is_reached # if not set the value to 0
        self.goal_reached += goal_is_reached # if it is add 1

        # Rewards
        if self.use_linear_rewards:
            velocity_reward = 1.0 / (1.0 + self.target_dist) * self.rew_scales["xy_velocity"]
            orient_reward = 1.0 / (1.0 + self.orient_dist) * self.rew_scales["heading"]
        elif self.use_square_rewards:
            velocity_reward = 1.0 / (1.0 + self.target_dist*self.target_dist) * self.rew_scales["xy_velocity"]
            orient_reward = 1.0 / (1.0 + self.orient_dist*self.orient_dist) * self.rew_scales["heading"]
        elif self.use_exponential_rewards:
            velocity_reward = torch.exp(-self.target_dist / 0.25) * self.rew_scales["xy_velocity"]
            orient_reward = torch.exp(-self.orient_dist / 0.25) * self.rew_scales["heading"]
        else:
            raise ValueError("Unknown reward type.")
        
        self.rew_buf[:] = velocity_reward + orient_reward
        # log episode reward sums
        self.episode_sums["xy_velocity_reward"] += velocity_reward
        self.episode_sums["heading"] += orient_reward
        # log raw info
        self.episode_sums["xy_velocity_error"] += self.target_dist
        self.episode_sums["heading"] += self.orient_dist

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > self._reset_error, ones, die)
        die = torch.where(self.root_dist > self._max_distance, ones, die)
        die = torch.where(self.goal_reached > self.kill_after_n_steps_in_tolerance, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
