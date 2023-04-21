
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
import torch
from gym import spaces

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class ModularFloatingPlatformCrippledTask(RLTask):
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
        # _num_quantized_actions has to be N whwere the final action space is N*2 +1
        self._num_quantized_actions = self._task_cfg["env"]["numQuantizedActions"]
        self.mass = self._task_cfg["env"]["mass"]
        self.thrust_force = self._task_cfg["env"]["thrustForce"]
        self.dt = self._task_cfg["sim"]["dt"]
        self.action_delay = self._task_cfg["env"]["actionDelay"]
        # subtasks legend: 0 - reach_zero, 1 - reach_target, 2 - reach_target & orientation, 
        #                  3 - reach_target & orientation & velocity
        self.subtask = self._task_cfg["env"]["subtask"]

        # define action space
        if self._discrete_actions=="MultiDiscrete":    
            self._num_actions = 8
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)])
            #self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
        elif self._discrete_actions=="Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        elif self._discrete_actions=="Quantised":
            self._num_actions = 8
        else:
            self._num_actions = 8

        self._num_observations = 18 + self._num_actions

        self._fp_position = torch.tensor([0, 0., 0.5])
        self._ball_position = torch.tensor([0, 0, 1.0])
        self._reset_dist = 5.

        # call parent class’s __init__
        RLTask.__init__(self, name, env)

        self.thrust_max = torch.tensor(self.thrust_force, device=self._device, dtype=torch.float32)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device, dtype=torch.float32)
        self.thrusters_state = torch.ones([self._num_envs, self._num_actions], device=self._device, dtype=torch.float32)

        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
     
        # Extra info
        self.extras = {}

        torch_zeros = lambda: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"rew_pos": torch_zeros(), "raw_dist": torch_zeros()}

        return

    def set_up_scene(self, scene) -> None:

        self.get_floating_platform()
        self.get_target()
        RLTask.set_up_scene(self, scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired

        root_path = "/World/envs/.*/Modular_floating_platform" 
        self._platforms = ModularFloatingPlatformView(prim_paths_expr=root_path, name="modular_floating_platform_view") 
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball")
        # set fp base masses according to the task config
        masses = torch.tensor(self.mass, device=self._device, dtype=torch.float).repeat(self.num_envs)
        self._platforms.base.set_masses(masses)

        # Add views to scene
        scene.add(self._platforms) # add view to scene for initialization
        scene.add(self._balls)
        for i in range(len(self._platforms.thrusters)):
            scene.add(self._platforms.thrusters[i])
        
        space_margin = " "*25
        print("\n########################  Floating platform set up ####################### \n")
        print(f'{space_margin} Number of thrusters: {len(self._platforms.thrusters)}')
        print(f'{space_margin} Mass base: {self._platforms.base.get_masses()[0]:.2f} kg')
        for i in range(len(self._platforms.thrusters)):
            # self._platforms.thrusters[i].set_masses(torch.tensor([50, 5], device=self._device))
            print(f'{space_margin} Mass thruster {i+1}: {self._platforms.thrusters[i].get_masses()[0]:.2f} kg')
        print(f'{space_margin} Thrust force: {self.thrust_force} N')
        print("\n##########################################################################")
        #stage = omni.usd.get_context().get_stage()
        #stage.Export("test_sim.usd")
        return

    def get_floating_platform(self):
        fp = ModularFloatingPlatform(prim_path=self.default_zero_env_path + "/Modular_floating_platform", name="modular_floating_platform",
                            translation=self._fp_position)
        
        self._sim_config.apply_articulation_settings("modular_floating_platform", get_prim_at_path(fp.prim_path),
                                                        self._sim_config.parse_actor_config("modular_floating_platform"))

    def get_target(self):
        radius = 0.2
        color = torch.tensor([1, 0, 0])
        ball = DynamicSphere(
            prim_path=self.default_zero_env_path + "/ball",
            translation=self._ball_position,
            name="target_0",
            radius=radius,
            color=color)
        self._sim_config.apply_articulation_settings("ball", get_prim_at_path(ball.prim_path),
                                                        self._sim_config.parse_actor_config("ball"))
        ball.set_collision_enabled(False)

    def get_observations(self) -> dict:
        # implement logic to retrieve observation states
        self.root_pos, self.root_rot = self._platforms.get_world_poses(clone=False)
        self.root_velocities = self._platforms.get_velocities(clone=False)
        root_positions = self.root_pos - self._env_pos
        root_quats = self.root_rot

        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)

        root_linvels = self.root_velocities[:, :3]
        root_angvels = self.root_velocities[:, 3:]

        self.obs_buf[..., 0:3] = self.target_positions - root_positions
        self.obs_buf[..., 3:6] = rot_x
        self.obs_buf[..., 6:9] = rot_y
        self.obs_buf[..., 9:12] = rot_z

        self.obs_buf[..., 12:15] = root_linvels
        self.obs_buf[..., 15:18] = root_angvels
        self.obs_buf[..., 18:18+self._num_actions] = self.thrusters_state

        observations = {
            self._platforms.name: {
               "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        # implement logic to be performed before physics steps

        if not self._env._world.is_playing():
            return        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        set_target_ids = (self.progress_buf % 500 == 0).nonzero(as_tuple=False).squeeze(-1)
        if len(set_target_ids) > 0:
            self.set_targets(set_target_ids)

        actions = actions.clone().to(self._device)
        self.actions = actions

        if  self._discrete_actions=="MultiDiscrete":
            # convert actions from [0, 1, 2] to [-1, 0, 1] to real force commands
            thrust_cmds = torch.sub(self.actions, 1.) 
        elif self._discrete_actions=="Quantised":
            # clamp to [-1.0, 1.0] the continuos actions
            thrust_cmds = torch.clamp(self.actions, min=0.0, max=1.0)
            thrust_cmds = quantize_tensor_values(thrust_cmds, self._num_quantized_actions).to(self._device)
        else:  
            # clamp to [-1.0, 1.0] the continuos actions
            thrust_cmds = torch.clamp((self.actions+1)/2, min=0.0, max=1.0)
            # write a mapping for the continuos actions to N quantised actions (N=20)

        thrusts = self.thrust_max * thrust_cmds * self.thrusters_state
        # thrusts given rotation
        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), 1).reshape(-1, 3, 3)
        rot_matrix = rot_matrix.repeat_interleave(self._num_actions, dim=0)
        force_x = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float32, device=self._device)
        force_y = torch.zeros(self._num_envs, self._num_actions, dtype=torch.float32, device=self._device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, self._num_actions, 2)
        thrusts = thrusts.reshape(-1, self._num_actions, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)
        thrusts = thrusts.reshape(-1, 3)
        self.thrusts = torch.matmul(rot_matrix, thrusts[:,:,None]).squeeze().reshape(self._num_envs, self.num_actions, 3)
        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # Apply action delay if required
        if self.action_delay > 0:
            time.sleep(self.action_delay)
        # Apply forces
        for i in range(self._num_actions):
            self._platforms.thrusters[i].apply_forces(self.thrusts[:,i], indices=self.all_indices, is_global=False)

    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._platforms.get_world_poses()
        self.root_velocities = self._platforms.get_velocities()
        self.dof_pos = self._platforms.get_joint_positions()
        self.dof_vel = self._platforms.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, self._num_actions, 3), dtype=torch.float32, device=self._device)
        
        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()

        if self.subtask == 0: # reach_zero
            # set target position randomly with x, y in (0, 0) and z in (2)
            self.target_positions[envs_long, 0:2] = torch.zeros((num_sets, 2), device=self._device)
        elif self.subtask == 1: # reach_target
            # set target position randomly with x, y in (-reset_dist, reset_dist) and z in (2)
            self.target_positions[envs_long, 0:2] = torch_rand_float(-self._reset_dist, self._reset_dist, (num_sets, 2), device=self._device) 
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        
        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._platforms.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        idx = torch.randint(0, self._num_actions, [num_resets], device=self._device)
        self.thrusters_state[env_ids,:] = 1 - torch.nn.functional.one_hot(idx, self._num_actions).float()

        # TODO: make sure resets are coherent for all different subtasks
        reset_pos = self._reset_dist if self.subtask == 0 else 0.0
        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-reset_pos, reset_pos, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-reset_pos, reset_pos, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 2] += torch_rand_float(-0.0, 0.0, (num_resets, 1), device=self._device).view(-1)
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        self._platforms.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        self._platforms.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        self._platforms.set_world_poses(root_pos[env_ids], self.initial_root_rot[env_ids].clone(), indices=env_ids)
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
        root_quats = self.root_rot 
        root_angvels = self.root_velocities[:, 3:]
        penalty = torch.square(root_angvels[:,-1])*0#(torch.abs(root_angvels[:,-1]) > 1.57) * (torch.abs(root_angvels[:,-1]) - 1.57) * 0
        #print(penalty[:5])
        # pos reward
        target_dist = torch.sqrt(torch.square(self.target_positions[:,:2] - root_positions[:,:2]).sum(-1))
        pos_reward = 1.0 / (1.0 + target_dist)
        self.target_dist = target_dist
        self.root_positions = root_positions

        # orinetation reward
        orient_z = torch.cos(root_quats[:, 0]) * torch.sin(root_quats[:, 1]) * torch.cos(root_quats[:, 2]) + torch.sin(root_quats[:, 0]) * torch.cos(root_quats[:, 1]) * torch.sin(root_quats[:, 2])
        self.orient_z = orient_z
        orient_reward = torch.where(orient_z > 0.0, torch.ones_like(orient_z), torch.zeros_like(orient_z))
        self.orient_reward = orient_reward
        #print(f'orient_reward: {orient_reward[0]}')
        #print(f'pos_reward: {pos_reward[0]}')
            # combined reward
        self.rew_buf[:] = pos_reward - penalty * 0.01
        #print(self.rew_buf[:5])
        # log episode reward sums
        self.episode_sums["rew_pos"] += pos_reward - penalty * 0.01
        # log raw info
        self.episode_sums["raw_dist"] += target_dist * 0.1

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        if self.subtask == 0:
            die = torch.where(self.target_dist > self._reset_dist * 2, ones, die) # die if going too far from target
        elif self.subtask == 1:
            die = torch.where(self.target_dist > self._reset_dist * 4, ones, die)
        # z >= 0.5 & z <= 5.0 & up > 0
        # die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)
        # die = torch.where(self.root_positions[..., 2] > 5.0, ones, die)
        # die = torch.where(self.orient_z < 0.0, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)