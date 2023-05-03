
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.MFP2D import ModularFloatingPlatform, compute_num_actions
from omniisaacgymenvs.robots.articulations.views.modular_floating_platform_view import ModularFloatingPlatformView
from omniisaacgymenvs.utils.pin import VisualPin

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import omni
import time
import math
import torch
from gym import spaces

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class MFP2DGoToXYDictRPTask(RLTask):
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
        self.kill_after_n_steps_in_tolerance = self._task_cfg["env"]["task_parameters"]["KillAfterNStepsInTolerance"]
        self._max_random_goal_position = self._task_cfg["env"]["task_parameters"]["MaxRandomGoalPosition"]
        self._max_reset_dist           = self._task_cfg["env"]["task_parameters"]["MaxResetDist"]
        self._min_reset_dist           = self._task_cfg["env"]["task_parameters"]["MinResetDist"]
        self._kill_dist                = self._task_cfg["env"]["task_parameters"]["KillDist"]
        self._num_actions              = self._task_cfg["env"]["task_parameters"]["MaxActions"]
        self._min_actions              = self._task_cfg["env"]["task_parameters"]["MinActions"]

        # Rewards parameters
        self.rew_scales = {}
        self.rew_scales["position"] = self._task_cfg["env"]["learn"]["PositionXYRewardScale"]
        self.rew_scales["position_exp_coeff"] = self._task_cfg["env"]["learn"]["PositionXYRewardExponentialCoefficient"]
        self.use_linear_rewards = self._task_cfg["env"]["learn"]["UseLinearRewards"]
        self.use_square_rewards = self._task_cfg["env"]["learn"]["UseSquareRewards"]
        self.use_exponential_rewards = self._task_cfg["env"]["learn"]["UseExponentialRewards"]

        self._num_observations = 18
        self._max_actions = compute_num_actions(self._platform_cfg)

        self.observation_space = spaces.Dict({"state":spaces.Box(np.ones(self.num_observations) * -np.Inf, np.ones(self.num_observations) * np.Inf),
                                              "transforms":spaces.Box(low=-1, high=1, shape=(self._num_actions, 4*4)),
                                              "masks":spaces.Box(low=0, high=1, shape=(self.num_actions,))})
        
        # define action space
        if self._discrete_actions=="MultiDiscrete":    
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)]*self._num_actions)
        elif self._discrete_actions=="Continuous":
            pass
        elif self._discrete_actions=="Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError("The requested discrete action type is not supported.")

        self._fp_position = torch.tensor([0, 0., 0.5])
        self._ball_position = torch.tensor([0, 0, 1.0])

        RLTask.__init__(self, name, env)

        self.thrust_max = torch.tensor(self.thrust_force, device=self._device, dtype=torch.float32)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self.transforms = torch.zeros((self._max_actions+1, 16), device=self._device, dtype=torch.float32)
        self.current_transforms = torch.zeros((self._num_envs, self._num_actions, 16), device=self._device, dtype=torch.float32)
        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device, dtype=torch.float32)
        self.thrusters_state = torch.ones([self._num_envs, self._max_actions], device=self._device, dtype=torch.long)
        self.action_masks = torch.ones([self._num_envs, self._num_actions], device=self._device, dtype=torch.long)
        self.sorted_thruster_indices = torch.zeros((self._num_envs, self._num_actions), device=self._device, dtype=torch.long)
        self.goal_reached = torch.zeros((self._num_envs), device=self._device, dtype=torch.int32)
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
     
        # Extra info
        self.extras = {}

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"position_reward": torch_zeros(), "position_error": torch_zeros()}
        return
    
    def cleanup(self) -> None:
        """ Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = {'state':torch.zeros((self._num_envs, self.num_observations), device=self._device, dtype=torch.float),
                        'transforms':torch.zeros((self._num_envs, self._num_actions, 16), device=self._device, dtype=torch.float),
                        'masks':torch.zeros((self._num_envs, self._num_actions), device=self._device, dtype=torch.float)}
                        #'thruster_state':torch.zeros((self._num_envs, self.num_actions), device=self._device, dtype=torch.int)}
        self.states_buf = torch.zeros((self._num_envs, self.num_states), device=self._device, dtype=torch.float)
        self.rew_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.float)
        self.reset_buf = torch.ones(self._num_envs, device=self._device, dtype=torch.long)
        self.progress_buf = torch.zeros(self._num_envs, device=self._device, dtype=torch.long)
        self.extras = {}

    def set_up_scene(self, scene) -> None:
        self.get_floating_platform()
        self.get_target()

        RLTask.set_up_scene(self, scene) 

        root_path = "/World/envs/.*/Modular_floating_platform" 
        self._platforms = ModularFloatingPlatformView(prim_paths_expr=root_path, name="modular_floating_platform_view") 
        self._pins = XFormPrimView(prim_paths_expr="/World/envs/.*/pin")

        # Add views to scene
        scene.add(self._platforms) # add view to scene for initialization
        scene.add(self._pins)
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
        self.transforms[:-1] = torch.from_numpy(fp._transforms).reshape([-1,4*4])
        self.transforms[-1] = torch.zeros([16])
        self._sim_config.apply_articulation_settings("modular_floating_platform", get_prim_at_path(fp.prim_path),
                                                        self._sim_config.parse_actor_config("modular_floating_platform"))

    def get_target(self):
        ball_radius = 0.2
        poll_radius = 0.025
        poll_length = 2
        color = torch.tensor([1, 0, 0])
        VisualPin(
            prim_path=self.default_zero_env_path + "/pin",
            translation=self._ball_position,
            name="target_0",
            ball_radius = ball_radius,
            poll_radius = poll_radius,
            poll_length = poll_length,
            color=color)

    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        self.root_pos, self.root_rot = self._platforms.get_world_poses(clone=False)
        self.root_velocities = self._platforms.get_velocities(clone=False)
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot
        # Get distance to the goal
        self.obs_buf["state"][..., 0:3] = self.target_positions - root_positions
        # Get rotation matrix from quaternions
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        self.obs_buf["state"][..., 3:6] = rot_x
        self.obs_buf["state"][..., 6:9] = rot_y
        self.obs_buf["state"][..., 9:12] = rot_z
        # Get velocities in the world frame 
        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]
        self.obs_buf["state"][..., 12:15] = root_linvels
        self.obs_buf["state"][..., 15:18] = root_angvels

        # Get thruster transforms
        self.obs_buf["transforms"] = self.current_transforms
        #print(self.current_transforms)
        #print(self.current_transforms[0])
        self.obs_buf["masks"] = self.action_masks

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
        thrusts = self.map_actions_to_thrusters(thrusts)
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
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self._device)
        self.initial_pin_rot[:, 0] = 1

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, self._max_actions, 3), dtype=torch.float32, device=self._device)
        
        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # Randomizes the position of the ball on the x y axis
        self.target_positions[envs_long, 0:2] = torch_rand_float(-self._max_random_goal_position, self._max_random_goal_position, (num_sets, 2), device=self._device)
        # Shifts the target up so it visually aligns better
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        # Projects to each environment
        pin_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        # Apply the new goals
        self._pins.set_world_poses(pin_pos[:, 0:3], self.initial_pin_rot[envs_long].clone(), indices=env_ids)

    def randomize_thruster_state(self, env_ids, num_envs):
        # Collects as many thrusters as their are actions
        weights = torch.ones(self._max_actions, device=self._device).expand(num_envs, -1)
        selected_thrusters = torch.multinomial(weights, num_samples=self._num_actions, replacement=False)
        if self._num_actions == self._max_actions:
            self.sorted_thruster_indices[env_ids] = selected_thrusters
        else:
            # Generates ones and zeros in uneven proportions across the batch
            max_kills = self._num_actions - self._min_actions
            weights = torch.ones((num_envs, 2), device=self._device)#.expand(num_envs, -1).clone()
            alpha = torch.rand((num_envs), device=self._device)
            weights[:,0] = alpha.clone()
            weights[:,1] = 1 - alpha.clone()
            idx2 = torch.multinomial(weights, num_samples=max_kills, replacement=True)
            # Selects L indices to set to N+1
            weights = torch.ones(self._num_actions, device=self._device).expand(num_envs, -1)
            idx3 = torch.multinomial(weights, num_samples=max_kills, replacement=False)
            # Creates a mask from both:
            idx4 = idx2*idx3 + (1 - idx2)*self._max_actions
            mask = torch.sum(torch.nn.functional.one_hot(idx4, self._max_actions+1),dim=1)
            # Removes the duplicates
            mask = mask[:,:self._num_actions]
            # Apply mask and add N+1
            final_idx = selected_thrusters * (1 - mask) + mask*(self._max_actions)
            # Sort such that the non-functional thrusters are at the end of the configuration
            _, sorted_idx = mask.sort(1)
            self.sorted_thruster_indices[env_ids] = torch.gather(final_idx, 1, sorted_idx)#.to(self._device)

    def map_actions_to_thrusters(self, actions):
        #print(actions.shape)
        return torch.zeros((self._num_envs, self._max_actions+1),device=self._device).scatter(1, self.sorted_thruster_indices, actions)[:,:self._max_actions]

    def update_transforms(self):
        # mem-free transforms
        idxs = self.sorted_thruster_indices.view(-1,self.num_actions,1).expand(-1,-1,16)
        trfs = self.transforms.view(1,-1, 16).expand(self.num_envs, -1, -1)
        self.current_transforms = torch.gather(trfs, 1, idxs)
        self.action_masks = self.sorted_thruster_indices == self._max_actions

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.goal_reached[env_ids] = 0
        self.randomize_thruster_state(env_ids, num_resets)
        self.update_transforms()
        # Randomizes the starting position of the platform within a disk around the target
        root_pos = self.initial_root_pos.clone()
        r = self._min_reset_dist + torch.rand((num_resets,), device=self._device) * (self._max_reset_dist - self._max_reset_dist)
        theta = torch.rand((num_resets,), device=self._device) * 2 * math.pi
        root_pos[env_ids, 0] += (r)*torch.cos(theta) + self.target_positions[env_ids, 0]
        root_pos[env_ids, 1] += (r)*torch.sin(theta) + self.target_positions[env_ids, 1]
        root_pos[env_ids, 2] += 0
        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._platforms.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0
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

        root_positions = self.root_pos - self._env_pos
        
        # position error
        self.target_dist = torch.sqrt(torch.square(self.target_positions[:,:2] - root_positions[:,:2]).sum(-1))
        self.root_positions = root_positions

        # Checks if the goal is reached
        goal_is_reached = (self.target_dist < self.xy_tolerance).int()
        self.goal_reached *= goal_is_reached # if not set the value to 0
        self.goal_reached += goal_is_reached # if it is add 1

        # Rewards
        if self.use_linear_rewards:
            position_reward = 1.0 / (1.0 + self.target_dist) * self.rew_scales["position"]
        elif self.use_square_rewards:
            position_reward = 1.0 / (1.0 + self.target_dist*self.target_dist) * self.rew_scales["position"]
        elif self.use_exponential_rewards:
            position_reward = torch.exp(-self.target_dist / self.rew_scales["position_exp_coeff"]) * self.rew_scales["position"]
        else:
            raise ValueError("Unknown reward type.")
        
        self.rew_buf[:] = position_reward
        # log episode reward sums
        self.episode_sums["position_reward"] += position_reward
        # log raw info
        self.episode_sums["position_error"] += self.target_dist

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > self._kill_dist, ones, die)
        die = torch.where(self.goal_reached > self.kill_after_n_steps_in_tolerance, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
