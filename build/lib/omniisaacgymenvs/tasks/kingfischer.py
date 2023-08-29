from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.MFP2D_virtual_thrusters import ModularFloatingPlatform
from omniisaacgymenvs.robots.articulations.views.mfp2d_virtual_thrusters_view import ModularFloatingPlatformView
from omniisaacgymenvs.utils.pin import VisualPin
from omniisaacgymenvs.utils.arrow import VisualArrow

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_thruster_generator import VirtualPlatform
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_task_factory import task_factory
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_core import parse_data_dict
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_task_rewards import Penalties
from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_disturbances import UnevenFloorDisturbance, NoisyObservations, NoisyActions

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.prims import get_prim_at_path

#buoyancy
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omniisaacgymenvs.envs.BuoyancyPhysics.Buoyancy_physics import *
from omniisaacgymenvs.envs.BuoyancyPhysics.ThrusterDynamics import *
from omniisaacgymenvs.envs.BuoyancyPhysics.Hydrodynamics import *


import numpy as np
import omni
import time
import math
import torch
from gym import spaces
from dataclasses import dataclass

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)

class KingfischerTask(RLTask):
    """
    The main class used to run tasks on the floating platform.
    Unlike other class in this repo, this class can be used to run different tasks.
    The idea being to extend it to multitask RL in the future."""

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
        #self._platform_cfg = self._task_cfg["env"]["platform"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self._device = self._cfg["sim_device"]
        self.step = 0

        # Domain randomization and adaptation
        #self.UF = UnevenFloorDisturbance(self._task_cfg, self._num_envs, self._device)
        self.ON = NoisyObservations(self._task_cfg)
        self.AN = NoisyActions(self._task_cfg)
        # Collects the platform parameters
        self.dt = self._task_cfg["sim"]["dt"]
        # Collects the task parameters
        task_cfg = self._task_cfg["env"]["task_parameters"]
        reward_cfg = self._task_cfg["env"]["reward_parameters"]
        penalty_cfg = self._task_cfg["env"]["penalties_parameters"]
        # Instantiate the task, reward and platform
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = parse_data_dict(Penalties(), penalty_cfg)
        #self.virtual_platform = VirtualPlatform(self._num_envs, self._platform_cfg, self._device)
        self._num_observations = self.task._num_observations
        self._max_actions = 2
        self._num_actions = 2

        #buoyancy

        #physics
        self.gravity=self._task_cfg["sim"]["gravity"][2]
        self.water_density=self._task_cfg["buoy"]["water_density"] # kg/m^3
        self.timeConstant = self._task_cfg["dynamics"]["thrusters"]["timeConstant"]

        #buoyancy
        self.average_buoyancy_force_value = self._task_cfg["dynamics"]["buoyancy"]["average_buoyancy_force_value"]
        self.amplify_torque = self._task_cfg["dynamics"]["buoyancy"]["amplify_torque"]

        #thrusters dynamics
            #interpolation
        self.cmd_lower_range = self._task_cfg["dynamics"]["thrusters"]["cmd_lower_range"]
        self.cmd_upper_range = self._task_cfg["dynamics"]["thrusters"]["cmd_upper_range"]
        self.numberOfPointsForInterpolation = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["numberOfPointsForInterpolation"]
        self.interpolationPointsFromRealData = self._task_cfg["dynamics"]["thrusters"]["interpolation"]["interpolationPointsFromRealData"]
            #least square methode
        self.neg_cmd_coeff=self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["neg_cmd_coeff"]
        self.pos_cmd_coeff=self._task_cfg["dynamics"]["thrusters"]["leastSquareMethod"]["pos_cmd_coeff"]
            #acceleration
        self.alpha = self._task_cfg["dynamics"]["acceleration"]["alpha"]
        self.last_time = self._task_cfg["dynamics"]["acceleration"]["last_time"]

        #boxes dimension to compute buoyancy forces and torques
        self.box_density=self._task_cfg["buoy"]["material_density"]
        self.box_width=self._task_cfg["buoy"]["box_width"]
        self.box_large=self._task_cfg["buoy"]["box_large"]
        self.box_high=self._task_cfg["buoy"]["box_high"]
        self.box_volume=self.box_width*self.box_large*self.box_high
        self.box_mass=self._task_cfg["buoy"]["mass"]
        self.half_box_size=self.box_high/2

        #damping constants
        self.squared_drag_coefficients = self._task_cfg["dynamics"]["damping"]["squared_drag_coefficients"]
        self.linear_damping = self._task_cfg["dynamics"]["damping"]["linear_damping"]
        self.quadratic_damping = self._task_cfg["dynamics"]["damping"]["quadratic_damping"]
        self.linear_damping_forward_speed = self._task_cfg["dynamics"]["damping"]["linear_damping_forward_speed"]
        self.offset_linear_damping = self._task_cfg["dynamics"]["damping"]["offset_linear_damping"]
        self.offset_lin_forward_damping_speed = self._task_cfg["dynamics"]["damping"]["offset_lin_forward_damping_speed"]
        self.offset_nonlin_damping = self._task_cfg["dynamics"]["damping"]["offset_nonlin_damping"]
        self.scaling_damping = self._task_cfg["dynamics"]["damping"]["scaling_damping"]
        self.offset_added_mass = self._task_cfg["dynamics"]["damping"]["offset_added_mass"]
        self.scaling_added_mass = self._task_cfg["dynamics"]["damping"]["scaling_added_mass"]


        RLTask.__init__(self, name, env)

        # Instantiate the action and observations spaces
        self.set_action_and_observation_spaces()
        # Sets the initial positions of the target and platform
        self._fp_position = torch.tensor([0, 0., 0.5])
        #self._default_marker_position = torch.tensor([0, 0, 1.0])
        #self._marker = None
        # Preallocate tensors
        self.actions = torch.zeros((self._num_envs, self._max_actions), device=self._device, dtype=torch.float32)
        self.heading = torch.zeros((self._num_envs, 2), device=self._device, dtype=torch.float32)
        self.all_indices = torch.arange(self._num_envs, dtype=torch.int32, device=self._device)
        # Extra info
        self.extras = {}
        # Episode statistics
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        self.add_stats(['normed_linear_vel', 'normed_angular_vel', 'actions_sum'])

        #buoyancy

        #positions constants
        self.left_thruster_position = torch.tensor(self._task_cfg["box"]["left_thruster_position"])
        self.right_thruster_position = torch.tensor(self._task_cfg["box"]["right_thruster_position"])

        #others positions constants that need to be GPU 
        self.boxes_initial_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.boxes_initial_rot = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.boxes_initial_pos[:, 2] = self.box_high/2
        self.boxes_initial_rot[:,0]=0.924    # y 45° rotation : 0.924
        self.boxes_initial_rot[:,2]=0.383    # y 45° rotation : 0.383
        self.boxes_initial_rot[:,1]=0.383


        #volume submerged
        self.high_submerged=torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)
        self.submerged_volume=torch.zeros((self._num_envs), device=self._device, dtype=torch.float32)

        #forces to be applied
        self.archimedes=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self.drag=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)
        self.thrusters=torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        #obs variables
        self.root_pos = torch.zeros((self._num_envs, 3), device=self._device, dtype=torch.float32)
        self.root_quats = torch.zeros((self._num_envs, 4), device=self._device, dtype=torch.float32)
        self.root_velocities = torch.zeros((self._num_envs, 6), device=self._device, dtype=torch.float32)

        ##some tests for the thrusters

        self.stop = torch.tensor([0.0, 0.0], device=self._device)
        self.turn = torch.tensor([-1.0, 0.5], device=self._device)
        self.forward = torch.tensor([1.0, 1.0], device=self._device)
        self.backward= - self.forward


        return
    
    def set_action_and_observation_spaces(self) -> None:
        """
        Sets the action and observation spaces."""

        # Defines the observation space
        self.observation_space = spaces.Dict({"state":spaces.Box(np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf),
                                              "transforms":spaces.Box(low=-1, high=1, shape=(self._max_actions, 5)),
                                              "masks":spaces.Box(low=0, high=1, shape=(self._max_actions,))})

        # Defines the action space
        if self._discrete_actions=="MultiDiscrete":    
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)]*self._max_actions)
        elif self._discrete_actions=="Continuous":
            pass
        elif self._discrete_actions=="Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError("The requested discrete action type is not supported.")

    
    def add_stats(self, names: list) -> None:
        """
        Adds training statistics to be recorded during training."""
        for name in names:
            torch_zeros = lambda: torch.zeros(self._num_envs, dtype=torch.float, device=self._device, requires_grad=False)
            if not name in self.episode_sums.keys():
                self.episode_sums[name] = torch_zeros()

    def cleanup(self) -> None:
        """
        Prepares torch buffers for RL data collection."""

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
        """
        Sets up the USD scene inside Omniverse for the task."""

        self.get_box()
        self.get_buoyancy()
        RLTask.set_up_scene(self, scene)
        self._boxes = RigidPrimView(prim_paths_expr="/World/envs/.*/box/body", name="box_view", reset_xform_properties=False)
        self._thrusters_left= RigidPrimView(prim_paths_expr="/World/envs/.*/box/left_thruster", name="left_thruster_view", reset_xform_properties=False)
        self._thrusters_right= RigidPrimView(prim_paths_expr="/World/envs/.*/box/right_thruster", name="right_thruster_view", reset_xform_properties=False)
        scene.add(self._boxes)
        scene.add(self._thrusters_left)
        scene.add(self._thrusters_right)

        # Add arrows to scene if task is go to pose
        #scene, self._marker = self.task.add_visual_marker_to_scene(scene)
        return

    def get_buoyancy(self):
        """create physics"""
        
        self.buoyancy_physics=BuoyantObject(self.num_envs, self._device, self.water_density, self.gravity, self.box_width/2, self.box_large/2, self.average_buoyancy_force_value, self.amplify_torque)
        self.thrusters_dynamics=DynamicsFirstOrder(self.num_envs, self._device, self.timeConstant, self.dt,self.numberOfPointsForInterpolation, self.interpolationPointsFromRealData, self.neg_cmd_coeff, self.pos_cmd_coeff, self.cmd_lower_range, self.cmd_upper_range )
        self.hydrodynamics=HydrodynamicsObject(self.num_envs, self._device, self.squared_drag_coefficients, self.linear_damping, self.quadratic_damping, self.linear_damping_forward_speed, self.offset_linear_damping, self.offset_lin_forward_damping_speed, self.offset_nonlin_damping, self.scaling_damping, self.offset_added_mass, self.scaling_added_mass, self.alpha,self.last_time )

    def get_box(self):
        """add to stage the usd file"""

        box_usd_path="/home/axelcoulon/projects/assets/box_thrusters.usd"
        box_prim_path=self.default_zero_env_path + "/box"
        add_reference_to_stage(prim_path=box_prim_path, usd_path=box_usd_path, prim_type="Xform")

    
    def update_state(self) -> None:
        """
        Updates the state of the system."""

        # Collects the position and orientation of the platform
        self.root_pos, self.root_quats = self._boxes.get_world_poses(clone=True)
        # Remove the offset from the different environments
        root_positions = self.root_pos - self._env_pos
        # Collects the velocity of the platform
        self.root_velocities = self._boxes.get_velocities(clone=True)
        root_velocities = self.root_velocities.clone()
        # Cast quaternion to Yaw
        siny_cosp = 2 * (self.root_quats[:,0] * self.root_quats[:,3] + self.root_quats[:,1] * self.root_quats[:,2])
        cosy_cosp = 1 - 2 * (self.root_quats[:,2] * self.root_quats[:,2] + self.root_quats[:,3] * self.root_quats[:,3])
        orient_z = torch.arctan2(siny_cosp, cosy_cosp)
        # Add noise on obs
        root_positions = self.ON.add_noise_on_pos(root_positions)
        root_velocities = self.ON.add_noise_on_vel(root_velocities)
        orient_z = self.ON.add_noise_on_heading(orient_z)
        # Compute the heading
        self.heading[:,0] = torch.cos(orient_z)
        self.heading[:,1] = torch.sin(orient_z)

        #get euler angles       
        self.euler_angles = self.get_euler_angles(self.root_quats) #rpy roll pitch yaws

        #body underwater
        self.high_submerged[:]=torch.clamp(self.half_box_size-self.root_pos[:,2], 0, self.box_high)
        self.submerged_volume[:]= torch.clamp(self.high_submerged * self.box_width * self.box_large, 0, self.box_volume)
        self.box_is_under_water = torch.where(self.high_submerged[:] > 0,1.0,0.0 ).unsqueeze(0)

        # Dump to state
        self.current_state = {"position":root_positions[:,:2], "orientation": self.heading, "linear_velocity": root_velocities[:,:2], "angular_velocity":root_velocities[:,-1]}


    def get_euler_angles(self, quaternions):
        """quaternions to euler"""

        angles = np.zeros((self.num_envs,3), dtype=float)
        for i in range(self._num_envs):
            angles[i,:]=quat_to_euler_angles(quaternions[i,:])
        return torch.tensor(angles).to(self._device)
    

    def get_observations(self) -> dict:
        """
        Gets the observations of the task to be passed to the policy."""

        # implement logic to retrieve observation states
        self.update_state()
        # Get the state
        self.obs_buf["state"] = self.task.get_state_observations(self.current_state)
        # Get thruster transforms
        self.obs_buf["transforms"] = 0.0 #self.virtual_platform.current_transforms
        # Get the action masks
        self.obs_buf["masks"] = 0.0     #self.virtual_platform.action_masks

        observations = {
            self._platforms.name: {
               "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        This function implements the logic to be performed before physics steps"""

        # If is not playing skip
        if not self._env._world.is_playing():
            return                
        # Check which environment need to be reset
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # Reset the environments (Robots)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        # Collect actions
        actions = actions.clone().to(self._device)
        self.actions = actions

        """ # Remap actions to the correct values
        if self._discrete_actions=="MultiDiscrete":
            # If actions are multidiscrete [0, 1]
            thrust_cmds = self.actions.float()
        elif self._discrete_actions=="Continuous":
            # Transform continuous actions to [0, 1] discrete actions.
            thrust_cmds = torch.clamp((self.actions+1)/2, min=0.0, max=1.0)
        else:
            raise NotImplementedError("") """
        
        self.thrusters_dynamics.set_target_force(self.forward) 

        # Apply forces
        self.apply_forces()
        return
    
    def apply_forces(self) -> None:
        """
        Applies all the forces to the platform and its thrusters."""

            ###archimedes
        self.archimedes[:,:]=self.buoyancy_physics.compute_archimedes_metacentric_local(self.submerged_volume, self.euler_angles, self.root_quats) * self.box_is_under_water[:,:].mT
            ###drag
        self.drag[:,:]=self.hydrodynamics.ComputeHydrodynamicsEffects(0.01, self.root_quats, self.root_velocities[:,:]) * self.box_is_under_water[:,:].mT
         
        self.thrusters[:,:] = self.thrusters_dynamics.update_forces()
        self.thrusters[:,:] *= self.box_is_under_water.mT
        
        #self.thruster_debugging_counter+=1

        self._boxes.apply_forces_and_torques_at_pos(forces=self.archimedes[:,:3] + self.drag[:,:3] , torques=self.archimedes[:,3:] + self.drag[:,3:], is_global=False)
        self._thrusters_left.apply_forces_and_torques_at_pos(self.thrusters[:,:3],positions=self.left_thruster_position,  is_global=False)
        self._thrusters_right.apply_forces_and_torques_at_pos(self.thrusters[:,3:], positions= self.right_thruster_position, is_global=False)
    
    
    def post_reset(self):
        """
        This function implements the logic to be performed after a reset."""

        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._boxes.get_world_poses()
        self.root_velocities = self._boxes.get_velocities()
    
        self.initial_root_pos, self.initial_root_rot = self.root_pos.clone(), self.root_rot.clone()
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self._device)
        self.initial_pin_rot[:, 0] = 1

        self.set_targets(self.all_indices)
        self.update_state()

    def set_targets(self, env_ids):
        """
        Sets the targets for the task."""

        num_sets = len(env_ids)
        env_long = env_ids.long()
        # Randomizes the position of the ball on the x y axis
        target_positions, target_orientation = self.task.get_goals(env_long, self.initial_pin_pos.clone(), self.initial_pin_rot.clone())
        target_positions[env_long, 2] = torch.ones(num_sets, device=self._device) * 2.0
        # Apply the new goals
        #if self._marker:
            #self._marker.set_world_poses(target_positions[env_long], target_orientation[env_long], indices=env_long)

    def set_to_pose(self, env_ids, positions, heading):
        """
        Sets the platform to a specific pose.
        TODO: Impose more iniiial conditions, such as linear and angular velocity."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        #self.virtual_platform.randomize_thruster_state(env_ids, num_resets)
        # Randomizes the starting position of the platform within a disk around the target
        root_pos = torch.zeros_like(self.root_pos)
        root_pos[env_ids,:2] = positions
        root_rot = torch.zeros_like(self.root_rot)
        root_rot[env_ids, :] = heading
        # Resets the states of the joints
        #self.dof_pos[env_ids, :] = torch.zeros((num_resets, self._platforms.num_dof), device=self._device)
        #self.dof_vel[env_ids, :] = 0
        # Sets the velocities to 0
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        #self._boxes.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        #self._platforms.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._boxes.set_world_poses(root_pos[env_ids], root_rot[env_ids], indices=env_ids)
        #self._boxes.set_velocities(root_velocities[env_ids], indices=env_ids)

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Resets the environments with the given indices."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        #self.virtual_platform.randomize_thruster_state(env_ids, num_resets)
        #self.UF.generate_floor(env_ids, num_resets)
        # Randomizes the starting position of the platform within a disk around the target
        root_pos, root_rot = self.task.get_spawns(env_ids, self.initial_root_pos.clone(), self.initial_root_rot.clone())
        # Resets the states of the joints
        #self.dof_pos[env_ids, :] = torch.zeros((num_resets, self._platforms.num_dof), device=self._device)
        #self.dof_vel[env_ids, :] = 0
        # Sets the velocities to 0
        root_velocities = self.root_velocities.clone()
        root_velocities[env_ids] = 0

        # apply resets
        #self._platforms.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)
        #self._platforms.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)
        self._boxes.set_world_poses(root_pos[env_ids], root_rot[env_ids], indices=env_ids)
        #self._boxes.set_velocities(root_velocities[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

        # fill `extras`
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"][key] = torch.mean(
                self.episode_sums[key][env_ids]) / self._max_episode_length
            self.episode_sums[key][env_ids] = 0.

    def update_state_statistics(self) -> None:
        """
        Updates the statistics of the state of the training."""

        self.episode_sums['normed_linear_vel'] += torch.norm(self.current_state["linear_velocity"], dim=-1)
        self.episode_sums['normed_angular_vel'] += torch.abs(self.current_state["angular_velocity"])
        self.episode_sums['actions_sum'] += torch.sum(self.actions, dim=-1)

    def calculate_metrics(self) -> None:
        """
        Calculates the metrics of the training.
        That is the rewards, penalties, and other perfomance statistics."""

        position_reward = self.task.compute_reward(self.current_state, self.actions)
        self.step += 1 / self._task_cfg["env"]["horizon_length"]
        penalties = self._penalties.compute_penalty(self.current_state, self.actions, self.step)
        self.rew_buf[:] = position_reward + penalties
        self.episode_sums = self.task.update_statistics(self.episode_sums)
        self.episode_sums = self._penalties.update_statistics(self.episode_sums)
        self.update_state_statistics()

    def is_done(self) -> None:
        """
        Checks if the episode is done."""

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = self.task.update_kills()

        # resets due to episode length
        self.reset_buf[:] = torch.where(self.progress_buf >= self._max_episode_length - 1, ones, die)