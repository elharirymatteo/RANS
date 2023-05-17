
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.MFP2D import ModularFloatingPlatform, compute_num_actions
from omniisaacgymenvs.robots.articulations.views.modular_floating_platform_view import ModularFloatingPlatformView
from omniisaacgymenvs.tasks.utils.fp_utils import quantize_tensor_values
from omniisaacgymenvs.utils.arrow import VisualArrow

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import math
import numpy as np
import omni
import time
import torch
from gym import spaces

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)


class MFP2DGoToPoseTask(RLTask):
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
        self._platform_cfg = self._task_cfg["env"]["platform"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["discreteActions"]

        # Platform parameters
        self.mass = self._task_cfg["env"]["mass"]
        self.thrust_force = self._task_cfg["env"]["thrustForce"]
        self.dt = self._task_cfg["sim"]["dt"]
        # Task parameters
        self.xy_tolerance = self._task_cfg["env"]["task_parameters"]["XYTolerance"]
        self.heading_tolerance = self._task_cfg["env"]["task_parameters"]["HeadingTolerance"]
        self.kill_after_n_steps_in_tolerance = self._task_cfg["env"]["task_parameters"]["KillAfterNStepsInTolerance"]
        self._max_random_goal_position = self._task_cfg["env"]["task_parameters"]["MaxRandomGoalPosition"]
        self._max_reset_dist           = self._task_cfg["env"]["task_parameters"]["MaxResetDist"]
        self._min_reset_dist           = self._task_cfg["env"]["task_parameters"]["MinResetDist"]
        self._kill_dist                = self._task_cfg["env"]["task_parameters"]["KillDist"]
        # Rewards parameters
        self.rew_scales = {}
        self.rew_scales["position"] = self._task_cfg["env"]["learn"]["PositionXYRewardScale"]
        self.rew_scales["heading"] = self._task_cfg["env"]["learn"]["HeadingRewardScale"]
        self.use_linear_rewards = self._task_cfg["env"]["learn"]["UseLinearRewards"]
        self.use_square_rewards = self._task_cfg["env"]["learn"]["UseSquareRewards"]
        self.use_exponential_rewards = self._task_cfg["env"]["learn"]["UseExponentialRewards"]

        # define action space
        self._num_actions = compute_num_actions(self._platform_cfg)
        if self._discrete_actions=="MultiDiscrete":    
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)]*self._num_actions)
        elif self._discrete_actions=="Continuous":
            pass
        elif self._discrete_actions=="Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError("The requested discrete action type is not supported.")
        self._num_observations = 20

        self._fp_position = torch.tensor([0, 0., 0.5])
        self._ball_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name, env)

        self.thrust_max = torch.tensor(self.thrust_force, device=self._device, dtype=torch.float32)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self.target_orient = torch.zeros((self._num_envs, 1), device=self._device, dtype=torch.float32)
        self.transforms = torch.zeros((self._num_actions, 4, 4), device=self._device, dtype=torch.float32)
        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device, dtype=torch.float32)
        self.goal_reached = torch.zeros((self.num_envs), device=self._device, dtype=torch.int32)
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
     
        # Extra info
        self.extras = {}

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"position_reward": torch_zeros(),"heading_reward":torch_zeros(), "position_error": torch_zeros(), "heading_error": torch_zeros()}
        return

    def set_up_scene(self, scene) -> None:
        self.get_floating_platform()
        self.get_target()
        RLTask.set_up_scene(self, scene)

        root_path = "/World/envs/.*/Modular_floating_platform" 
        self._platforms = ModularFloatingPlatformView(prim_paths_expr=root_path, name="modular_floating_platform_view") 
        self._arrows = XFormPrimView(prim_paths_expr="/World/envs/.*/arrow")

        # Add views to scene
        scene.add(self._platforms) # add view to scene for initialization
        scene.add(self._arrows)
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
                            translation=self._fp_position, cfg=self._platform_cfg)
        self.transforms[...] = torch.from_numpy(fp._transforms)
        
        self._sim_config.apply_articulation_settings("modular_floating_platform", get_prim_at_path(fp.prim_path),
                                                        self._sim_config.parse_actor_config("modular_floating_platform"))

    def get_target(self):
        body_radius = 0.1
        body_length = 0.5
        poll_radius = 0.025
        poll_length = 2
        head_radius = 0.2
        head_length = 0.5
        color = torch.tensor([1, 0, 0])
        arrow = VisualArrow(
            prim_path=self.default_zero_env_path + "/arrow",
            translation=self._ball_position,
            name="target_0",
            body_radius=body_radius,
            body_length=body_length,
            poll_radius=poll_radius,
            poll_length=poll_length,
            head_radius=head_radius,
            head_length=head_length,
            color=color)

    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        self.root_pos, self.root_rot = self._platforms.get_world_poses(clone=False)
        self.root_velocities = self._platforms.get_velocities(clone=False)
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        # Get distance to the goal
        self.obs_buf[..., 0:3] = self.target_positions - root_positions
        # Get rotation matrix from quaternions
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z
        # Get velocities in the world frame  
        self.obs_buf[..., 12:15] = self.root_velocities[:, :3]
        self.obs_buf[..., 15:18] = self.root_velocities[:, 3:]
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
        
        # Applies thrust multiplier
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
        self.initial_arrow_pos = self._env_pos
        self.initial_arrow_rot = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self._device)
        self.initial_arrow_rot[:, 0] = 1

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, self._num_actions, 3), dtype=torch.float32, device=self._device)
        
        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # Randomizes the position of the target
        self.target_positions[envs_long, 0:2] = torch_rand_float(-self._max_random_goal_position, self._max_random_goal_position, (num_sets, 2), device=self._device)
        # Shifts the target up so it visually aligns better
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        # Projects to each environment
        arrow_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        # Randomizes the heading of the target
        self.target_orient[envs_long, 0] = torch.rand(num_sets, device=self._device) * math.pi
        self.initial_arrow_rot[envs_long, 0:1] = torch.cos(self.target_orient[envs_long]*0.5)
        self.initial_arrow_rot[envs_long, 3:4] = torch.sin(self.target_orient[envs_long]*0.5)
        # Apply the new goals (arrows)
        self._arrows.set_world_poses(arrow_pos[:, 0:3], self.initial_arrow_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.goal_reached[env_ids] = 0
        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._platforms.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0
        # Randomizes the starting position of the platform
        root_pos = self.initial_root_pos.clone()
        r = self._min_reset_dist + torch.rand((num_resets,), device=self._device) * (self._max_reset_dist - self._min_reset_dist)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        root_pos[env_ids, 0] += (r)*torch.cos(theta) + self.target_positions[env_ids, 0]
        root_pos[env_ids, 1] += (r)*torch.sin(theta) + self.target_positions[env_ids, 1]
        root_pos[env_ids, 2] += 0
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
            self.extras["episode"][key] = self.episode_sums[key]
            # self.extras["episode"][key] = torch.mean(
            #     self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.

    def calculate_metrics(self) -> None:
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot 
        # Cast quaternion to Yaw
        siny_cosp = 2 * (root_quats[:,0] * root_quats[:,3] + root_quats[:,1] * root_quats[:,2])
        cosy_cosp = 1 - 2 * (root_quats[:,2] * root_quats[:,2] + root_quats[:,3] * root_quats[:,3])
        orient_z = torch.arctan2(siny_cosp, cosy_cosp)
        
        # position error
        self.target_dist = torch.sqrt(torch.square(self.target_positions[:,:2] - root_positions[:,:2]).sum(-1))
        self.root_positions = root_positions

        # orientation error
        self.orient_z = orient_z
        self.orient_dist = torch.abs(torch.arctan2(torch.sin(self.target_orient[:,0] - orient_z), torch.cos(self.target_orient[:,0] - orient_z)))

        # Checks if the goal is reached
        position_goal_is_reached = (self.target_dist < self.xy_tolerance).int()
        heading_goal_is_reached = (self.orient_dist < self.heading_tolerance).int()
        goal_is_reached = position_goal_is_reached * heading_goal_is_reached
        self.goal_reached *= goal_is_reached # if not set the value to 0
        self.goal_reached += goal_is_reached # if it is add 1

        # Rewards
        if self.use_linear_rewards:
            position_reward = 1.0 / (1.0 + self.target_dist) * self.rew_scales["position"]
            heading_reward = 1.0 / (1.0 + self.orient_dist)  * self.rew_scales["heading"]
        elif self.use_square_rewards:
            position_reward = 1.0 / (1.0 + self.target_dist*self.target_dist) * self.rew_scales["position"]
            heading_reward = 1.0 / (1.0 + self.orient_dist*self.orient_dist)  * self.rew_scales["heading"]
        elif self.use_exponential_rewards:
            position_reward = torch.exp(-self.target_dist / 0.25) * self.rew_scales["position"]
            heading_reward = torch.exp(-self.orient_dist / 0.25)  * self.rew_scales["heading"]
        else:
            raise ValueError("Unknown reward type.")
        
        self.rew_buf[:] = position_reward + heading_reward
        # log episode reward sums
        self.episode_sums["position_reward"] += position_reward
        self.episode_sums["heading_reward"] += heading_reward
        # log raw info
        self.episode_sums["position_error"] += self.target_dist
        self.episode_sums["heading_error"] += self.orient_dist

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > self._kill_dist, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
