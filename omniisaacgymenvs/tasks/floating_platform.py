
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.floating_platform import FloatingPlatform
from omniisaacgymenvs.robots.articulations.views.floating_platform_view import FloatingPlatformView
from omniisaacgymenvs.tasks.utils.fp_utils import quantize_tensor_values

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch
from gym import spaces

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

# creating set of allowed thrust actions
DISCRETE_ACTIONS = torch.tensor([[0, 0, 0, 0], # no action
                        [1, 0, 0, 0],[0, 1, 0, 0], # single thrusts positive
                        [0, 0, 1, 0], [0, 0, 0, 1], 
                        [-1, 0, 0, 0],[0, -1, 0, 0], # single thrusts negative
                        [0, 0, -1, 0], [0, 0, 0, -1],
                        [1, 1, 0, 0],[-1, -1, 0, 0], # thr 1 & thr 2, move forward, backward on X axis
                        [0, 0, 1, 1],[0, 0, -1, -1], # thr 3 & thr 4, move forward, backward on Y axis
                        [1, 0, 1, 0], [-1, 0, -1, 0], # move diagonal  NE & SW (using t1-t3)
                        [1, 0, 0, -1], [-1, 0, 0, 1], # move diagonal  NW & SE (using t1-t4)                       
                        [1, -1, 0, 0], [-1, 1, 0, 0], # rotate clock/counter-clockwise
                        [0, 0, 1, -1], [0, 0, -1, 1]], # rotate clock/counter-clockwise
                        device="cuda") 

class FloatingPlatformTask(RLTask):
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

        self.dt = self._task_cfg["sim"]["dt"]
        
        self._num_observations = 18

        # define action space
        if self._discrete_actions=="MultiDiscrete":    
            self._num_actions = 4
            self.action_space = spaces.MultiDiscrete([3, 3, 3, 3])
        
        elif self._discrete_actions=="Discrete":
            self._num_actions = 21
            self.action_space = spaces.Discrete(self._num_actions)  

        elif self._discrete_actions=="Quantised":
            self._num_actions = 4     
            self._num_quantized_actions = 20
        
        else:
            self._num_actions = 4


        self._fp_position = torch.tensor([0, 0., 0.5])
        self._ball_position = torch.tensor([0, 0, 3.0])
        self._reset_dist = 5.

        # call parent class’s __init__
        RLTask.__init__(self, name, env)

        thrust_max = 40
        self.thrust_max = torch.tensor(thrust_max, device=self._device, dtype=torch.float32)

        self.target_positions = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.target_positions[:, 2] = 1
        self.actions = torch.zeros((self._num_envs, self._num_actions), device=self._device, dtype=torch.float32)

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

        root_path = "/World/envs/.*/Floating_platform" 
        self._platforms = FloatingPlatformView(prim_paths_expr=root_path, name="floating_platform_view") 
        self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball")
        print(len(self._platforms.thrusters))

        scene.add(self._platforms) # add view to scene for initialization
        scene.add(self._balls)
        for i in range(4):
            scene.add(self._platforms.thrusters[i])
        return

    def get_floating_platform(self):
        fp = FloatingPlatform(prim_path=self.default_zero_env_path + "/Floating_platform", name="floating_platform",
                            translation=self._fp_position)
        self._sim_config.apply_articulation_settings("floating_platform", get_prim_at_path(fp.prim_path),
                                                        self._sim_config.parse_actor_config("floating_platform"))

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
        #print(f'ACTIONS: {self.actions} \n TYPE:{self.actions.dtype}')
        #print(f' ACTIONS SHAPE : {self.actions.shape} NDIM: {self.actions.ndim}')

        ## DISCRETE ACTIONS MAPPING
         # the agents selectes for 4 thrusters, a value between [0,2]                       #
         # then this values are shifter left to have values centered in 0 (-1, 0, 1)        #
         # to which a thrust_max is multiplied, obtainig the final actinon to be delivered  #
        if  self._discrete_actions=="MultiDiscrete":
            # convert actions from [0, 1, 2] to [-1, 0, 1] to real force commands
            thrust_cmds = torch.sub(self.actions, 1.) 

        elif self._discrete_actions=="Discrete":
            self.actions = self.actions.squeeze(-1) if self.actions.ndim==2 else self.actions
            # print(f'ACTIONS: {self.actions} \n TYPE:{self.actions.dtype}, NDIM: {self.actions.ndim}')
            # get the allowed actions based on the agent discrete actions selected
            thrust_cmds = DISCRETE_ACTIONS.index_select(0, self.actions)
        elif self._discrete_actions=="Quantised":
            # clamp to [-1.0, 1.0] the continuos actions
            thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
            thrust_cmds = quantize_tensor_values(thrust_cmds, self._num_quantized_actions).to(self._device)
            # print(f'ACTIONS: {thrust_cmds} \n TYPE:{thrust_cmds.dtype}, NDIM: {thrust_cmds.ndim}')
        else:  
            # clamp to [-1.0, 1.0] the continuos actions
            thrust_cmds = torch.clamp(actions, min=-1.0, max=1.0)
            # write a mapping for the continuos actions to N quantised actions (N=20) 

        thrusts = self.thrust_max * thrust_cmds

        #print(f'Actions space: {self.action_space}')
        #print(f'thrusts: {thrusts}')

        # thrusts given rotation
        root_quats = self.root_rot
        rot_x = quat_axis(root_quats, 0)
        rot_y = quat_axis(root_quats, 1)
        rot_z = quat_axis(root_quats, 2)
        rot_matrix = torch.cat((rot_x, rot_y, rot_z), 1).reshape(-1, 3, 3)

        force_x = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_y = torch.zeros(self._num_envs, 4, dtype=torch.float32, device=self._device)
        force_xy = torch.cat((force_x, force_y), 1).reshape(-1, 4, 2)
        thrusts = thrusts.reshape(-1, 4, 1)
        thrusts = torch.cat((force_xy, thrusts), 2)        

        
        thrusts_0 = thrusts[:, 0]
        thrusts_0 = thrusts_0[:, :, None]

        thrusts_1 = thrusts[:, 1]
        thrusts_1 = thrusts_1[:, :, None]

        thrusts_2 = thrusts[:, 2]
        thrusts_2 = thrusts_2[:, :, None]

        thrusts_3 = thrusts[:, 3]
        thrusts_3 = thrusts_3[:, :, None]

        mod_thrusts_0 = torch.matmul(rot_matrix, thrusts_0)
        mod_thrusts_1 = torch.matmul(rot_matrix, thrusts_1)
        mod_thrusts_2 = torch.matmul(rot_matrix, thrusts_2)
        mod_thrusts_3 = torch.matmul(rot_matrix, thrusts_3)

        self.thrusts[:, 0] = torch.squeeze(mod_thrusts_0)
        self.thrusts[:, 1] = torch.squeeze(mod_thrusts_1)
        self.thrusts[:, 2] = torch.squeeze(mod_thrusts_2)
        self.thrusts[:, 3] = torch.squeeze(mod_thrusts_3)
        # clear actions for reset envs
        self.thrusts[reset_env_ids] = 0

        # Apply forces
        for i in range(4):
            self._platforms.thrusters[i].apply_forces(self.thrusts[:, i], indices=self.all_indices, is_global=False)

    def post_reset(self):
        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._platforms.get_world_poses()
        self.root_velocities = self._platforms.get_velocities()
        self.dof_pos = self._platforms.get_joint_positions()
        self.dof_vel = self._platforms.get_joint_velocities()

        self.initial_ball_pos, self.initial_ball_rot = self._balls.get_world_poses(clone=False)
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()

        # control parameters
        self.thrusts = torch.zeros((self._num_envs, 4, 3), dtype=torch.float32, device=self._device)
        
        self.set_targets(self.all_indices)

    def set_targets(self, env_ids):
        num_sets = len(env_ids)
        envs_long = env_ids.long()
        # set target position randomly with x, y in (0, 0) and z in (2)
        self.target_positions[envs_long, 0:2] = torch.zeros((num_sets, 2), device=self._device)
        self.target_positions[envs_long, 2] = torch.ones(num_sets, device=self._device) * 2.0

        # shift the target up so it visually aligns better
        ball_pos = self.target_positions[envs_long] + self._env_pos[envs_long]
        ball_pos[:, 2] += 0.0
        self._balls.set_world_poses(ball_pos[:, 0:3], self.initial_ball_rot[envs_long].clone(), indices=env_ids)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        self.dof_pos[env_ids, :] = torch_rand_float(-0.0, 0.0, (num_resets, self._platforms.num_dof), device=self._device)
        self.dof_vel[env_ids, :] = 0

        root_pos = self.initial_root_pos.clone()
        root_pos[env_ids, 0] += torch_rand_float(-self._reset_dist, self._reset_dist, (num_resets, 1), device=self._device).view(-1)
        root_pos[env_ids, 1] += torch_rand_float(-self._reset_dist, self._reset_dist, (num_resets, 1), device=self._device).view(-1)
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

        # pos reward
        target_dist = torch.sqrt(torch.square(self.target_positions - root_positions).sum(-1))
        pos_reward = 1.0 / (1.0 + target_dist)
        self.target_dist = target_dist
        self.root_positions = root_positions

            # combined reward
        self.rew_buf[:] = pos_reward 
        # log episode reward sums
        self.episode_sums["rew_pos"] += pos_reward
        # log raw info
        self.episode_sums["raw_dist"] += target_dist

    def is_done(self) -> None:
        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = torch.zeros_like(self.reset_buf)
        die = torch.where(self.target_dist > self._reset_dist * 2, ones, die) # die if going too far from target

        # z >= 0.5 & z <= 5.0 & up > 0
        # die = torch.where(self.root_positions[..., 2] < 0.5, ones, die)
        # die = torch.where(self.root_positions[..., 2] > 5.0, ones, die)
        # die = torch.where(self.orient_z < 0.0, ones, die)

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)
