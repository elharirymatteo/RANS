from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.MFP2D_virtual_thrusters import ModularFloatingPlatform
from omniisaacgymenvs.robots.articulations.views.mfp2d_virtual_thrusters_view import ModularFloatingPlatformView
from omniisaacgymenvs.utils.pin import VisualPin

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_go_to_xy import GoToXYTask
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_go_to_pose import GoToPoseTask

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import omni
import time
import math
import torch
from gym import spaces
from dataclasses import dataclass

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class MFP2DVirtual(RLTask):
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
        self._discrete_actions = self._task_cfg["env"]["action_mode"]

        # Platform parameters
        self.dt = self._task_cfg["sim"]["dt"]

        # Task parameters
        self._max_actions              = self._task_cfg["env"]["platform"]["max_thrusters"]
        self._min_actions              = self._task_cfg["env"]["platform"]["min_thrusters"]
        self.thrust_max              = self._task_cfg["env"]["platform"]["max_thrust_force"]
        self._num_actions              = self._max_actions

        self._num_observations = 1
        RLTask.__init__(self, name, env)
        self.task = GoToXYTask(self._task_cfg["env"]["task_parameters"], self._task_cfg["env"]["reward_parameters"], self._num_envs, self._device)
        self._num_observations = self.task._num_observations

        self.observation_space = spaces.Dict({"state":spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf),
                                              "transforms":spaces.Box(low=-1, high=1, shape=(self._max_actions, 5)),
                                              "masks":spaces.Box(low=0, high=1, shape=(self._max_actions,))})

        # define action space
        if self._discrete_actions=="MultiDiscrete":    
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)]*self._max_actions)
        elif self._discrete_actions=="Continuous":
            pass
        elif self._discrete_actions=="Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError("The requested discrete action type is not supported.")

        self._fp_position = torch.tensor([0, 0., 0.5])
        self._ball_position = torch.tensor([0, 0, 1.0])

        self.current_transforms = torch.zeros((self._num_envs, self._max_actions, 5), device=self._device, dtype=torch.float32)
        self.transforms2D = torch.zeros((self._num_envs, self._max_actions, 3, 3), device=self._device, dtype=torch.float32)
        self.action_masks = torch.ones([self._num_envs, self._max_actions], device=self._device, dtype=torch.long)
        self.actions = torch.zeros((self._num_envs, self._max_actions), device=self._device, dtype=torch.float32)
        self.heading = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
     
        # Extra info
        self.extras = {}

        self.episode_sums = self.task.create_stats({})
        return
    
    def cleanup(self) -> None:
        """ Prepares torch buffers for RL data collection."""

        # prepare tensors
        self.obs_buf = {'state':torch.zeros((self._num_envs, self._num_observations), device=self._device, dtype=torch.float),
                        'transforms':torch.zeros((self._num_envs, self._max_actions, 5), device=self._device, dtype=torch.float),
                        'masks':torch.zeros((self._num_envs, self._max_actions), device=self._device, dtype=torch.float)}

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
        scene.add(self._platforms)
        scene.add(self._pins)
        scene.add(self._platforms.thrusters)
        return

    def get_floating_platform(self):
        fp = ModularFloatingPlatform(prim_path=self.default_zero_env_path + "/Modular_floating_platform", name="modular_floating_platform",
                            translation=self._fp_position, cfg=self._platform_cfg)
        self._sim_config.apply_articulation_settings("modular_floating_platform", get_prim_at_path(fp.prim_path),
                                                        self._sim_config.parse_actor_config("modular_floating_platform"))

    def get_target(self) -> None:
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

    def update_state(self) -> None:
        root_pos, root_quats = self._platforms.get_world_poses(clone=False)
        root_velocities = self._platforms.get_velocities(clone=False)
        root_positions = root_pos - self._env_pos
        # Cast quaternion to Yaw
        siny_cosp = 2 * (root_quats[:,0] * root_quats[:,3] + root_quats[:,1] * root_quats[:,2])
        cosy_cosp = 1 - 2 * (root_quats[:,2] * root_quats[:,2] + root_quats[:,3] * root_quats[:,3])
        orient_z = torch.arctan2(siny_cosp, cosy_cosp)
        self.heading[:,0] = torch.cos(orient_z)
        self.heading[:,1] = torch.sin(orient_z)
        self.current_state = {"position":root_positions[:,:2], "orientation": self.heading, "linear_velocity": root_velocities[:,:2], "angular_velocity":root_velocities[:,-1]}

    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        self.update_state()
        # Get the state
        self.obs_buf["state"] = self.task.get_state_observations(self.current_state)
        # Get thruster transforms
        self.obs_buf["transforms"] = self.current_transforms
        # Get the action masks
        self.obs_buf["masks"] = self.action_masks

        print(self.obs_buf["state"][0])

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
        # clear actions for reset envs
        thrusts[reset_env_ids] = 0
        T = self.transforms2D.expand(self._num_envs, self._max_actions, 3, 3)
        I = torch.arange(self._num_envs*self._max_actions)

        R = T[:, :, :2,:2]
        O = T[:, :, 2,:2]
        z = torch.zeros((O.shape[0],self._max_actions,1), device = self._device)
        O = torch.cat([O,z], dim=-1)
        R = R.reshape(-1,2,2)
        O = O.reshape(-1,3)
        
        A = thrusts.reshape(-1,1)
        y = torch.zeros_like(thrusts.reshape(-1,1))
        A = torch.cat([A,y],dim=-1)
        A = torch.matmul(R, A.view(self._num_envs*self._max_actions,2,1))
        z = torch.zeros_like(thrusts.reshape(-1,1))
        A = torch.cat([A[:,:,0],z],dim=-1)

        # Apply forces
        self._platforms.thrusters.apply_forces_and_torques_at_pos(forces=A, positions=O, indices=I, is_global=False)
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
        env_long = env_ids.long()
        # Randomizes the position of the ball on the x y axis
        target_positions, target_orientation = self.task.get_goals(env_long, self.initial_pin_pos.clone(), self.initial_pin_rot.clone())
        target_positions[env_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        # Apply the new goals
        self._pins.set_world_poses(target_positions[env_long], target_orientation[env_long], indices=env_long)

    def randomize_thruster_state(self, env_ids, num_envs):
        random_offset = torch.ones((num_envs), device=self._device).view(-1,1).expand(num_envs, 8) * math.pi / 4
        thrust_offset = torch.arange(4, device=self._device).repeat_interleave(2).expand(num_envs, 8)/4 * math.pi * 2
        thrust_90 = (torch.arange(2, device=self._device).repeat(4).expand(num_envs,8) * 2 - 1) * math.pi / 2 
        mask = torch.ones((num_envs, 8), device=self._device)

        theta = random_offset + thrust_offset + thrust_90
        theta2 = random_offset + thrust_offset

        self.transforms2D[env_ids,:,0,0] = torch.cos(theta) * mask
        self.transforms2D[env_ids,:,0,1] = torch.sin(-theta) * mask
        self.transforms2D[env_ids,:,1,0] = torch.sin(theta) * mask
        self.transforms2D[env_ids,:,1,1] = torch.cos(theta) * mask
        self.transforms2D[env_ids,:,2,0] = torch.cos(theta2) * 0.5 * mask
        self.transforms2D[env_ids,:,2,1] = torch.sin(theta2) * 0.5 * mask
        self.transforms2D[env_ids,:,2,2] = 1 * mask

        self.action_masks[env_ids, :] = 1 - mask.long()

        self.current_transforms[env_ids, :, 0] = torch.cos(theta) * mask
        self.current_transforms[env_ids, :, 1] = torch.sin(-theta) * mask
        self.current_transforms[env_ids, :, 2] = torch.cos(theta2) * 0.5 * mask
        self.current_transforms[env_ids, :, 3] = torch.sin(theta2) * 0.5 * mask
        self.current_transforms[env_ids, :, 4] = 1 * mask

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        self.randomize_thruster_state(env_ids, num_resets)
        # Randomizes the starting position of the platform within a disk around the target
        root_pos, root_rot = self.task.get_spawns(env_ids, self.initial_root_pos.clone(), self.initial_root_rot.clone())
        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch.zeros((num_resets, self._platforms.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0
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
        position_reward, target_dist = self.task.compute_reward(self.current_state, self.actions)
        print(target_dist[0], position_reward[0])
        self.rew_buf[:] = position_reward
        self.episode_sums = self.task.update_statistics(self.episode_sums)

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = self.task.update_kills()

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
