__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.heron import (
    Heron,
)
from omniisaacgymenvs.robots.articulations.views.heron_view import (
    HeronView,
)
from omniisaacgymenvs.utils.pin import VisualPin
from omniisaacgymenvs.utils.arrow import VisualArrow

from omniisaacgymenvs.tasks.ASV.task_factory import (
    task_factory,
)
from omniisaacgymenvs.tasks.common_3DoF.penalties import (
    EnvironmentPenalties,
)
from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    Disturbances,
)

from omniisaacgymenvs.envs.Physics.Hydrodynamics import Hydrodynamics
from omniisaacgymenvs.envs.Physics.Hydrostatics import Hydrostatics
from omniisaacgymenvs.envs.Physics.ThrusterDynamics import DynamicsFirstOrder

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.prims import get_prim_at_path

from typing import Dict, List, Tuple

import numpy as np
import omni
import time
import math
import wandb
import torch
from gym import spaces
from dataclasses import dataclass

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class ASVVirtual(RLTask):
    """
    The main class used to run tasks on the floating platform.
    Unlike other class in this repo, this class can be used to run different tasks.
    The idea being to extend it to multitask RL in the future."""

    def __init__(
        self,
        name: str,  # name of the Task
        sim_config,  # SimConfig instance for parsing cfg
        env,  # env instance of VecEnvBase or inherited class
        offset=None,  # transform offset in World
    ) -> None:
        # parse configurations, set task-specific members
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        self._enable_wandb_logs = self._task_cfg["enable_wandb_log"]
        self._device = self._cfg["sim_device"]
        self._robot_cfg = self._task_cfg["robot"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self.iteration = 0
        self.step = 0

        # Collects the platform parameters
        self.dt = self._task_cfg["sim"]["dt"]
        # Collects the task parameters
        task_cfg = self._task_cfg["sub_task"]
        reward_cfg = self._task_cfg["reward"]
        penalty_cfg = self._task_cfg["penalty"]
        domain_randomization_cfg = self._task_cfg["disturbances"]

        # physics
        self.gravity = self._task_cfg["sim"]["gravity"][2]
        self.timeConstant = self._robot_cfg["dynamics"]["thrusters"]["timeConstant"]

        # hydrodynamics
        self.hydrodynamics_cfg = self._robot_cfg["dynamics"]["hydrodynamics"]

        # hydrostatics
        self.hydrostatics_cfg = self._robot_cfg["dynamics"]["hydrostatics"]

        # thrusters dynamics
        self.thrusters_dynamics_cfg = self._robot_cfg["dynamics"]["thrusters"]

        # Instantiate the task, reward and platform
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = EnvironmentPenalties(**penalty_cfg)
        self.DR = Disturbances(
            domain_randomization_cfg,
            num_envs=self._num_envs,
            device=self._device,
        )
        self._num_observations = self.task._num_observations * self.task._obs_buffer_len
        self._max_actions = 2  # Number of thrusters
        self._num_actions = 2  # Number of thrusters
        RLTask.__init__(self, name, env)
        # Instantiate the action and observations spaces
        self.set_action_and_observation_spaces()
        # Sets the initial positions of the target and platform
        self._asv_position = torch.tensor([0, 0.0, 0.0])
        self._default_marker_position = torch.tensor([0, 0, 1.0])
        self._marker = None
        # Preallocate tensors
        self.actions = torch.zeros(
            (self._num_envs, self._max_actions),
            device=self._device,
            dtype=torch.float32,
        )
        self.heading = torch.zeros(
            (self._num_envs, 2), device=self._device, dtype=torch.float32
        )
        self.all_indices = torch.arange(
            self._num_envs, dtype=torch.int32, device=self._device
        )
        # Extra info
        self.extras = {}
        # Episode statistics
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        # self.add_stats(["normed_linear_vel", "normed_angular_vel", "actions_sum"])

        # obs variables
        self.root_pos = torch.zeros(
            (self._num_envs, 3), device=self._device, dtype=torch.float32
        )
        self.root_quats = torch.zeros(
            (self._num_envs, 4), device=self._device, dtype=torch.float32
        )
        self.root_quats[:, 0] = 1.0
        self.root_velocities = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )

        # forces to be applied
        self.hydrostatic_force = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.drag = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.thrusters = torch.zeros(
            (self._num_envs, 6), device=self._device, dtype=torch.float32
        )
        self.contact_state = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )

        ##some tests for the thrusters

        self.stop = torch.tensor([0.0, 0.0], device=self._device)
        self.turn_right = torch.tensor([1.0, -1.0], device=self._device)
        self.turn_left = torch.tensor([-1.0, 1.0], device=self._device)
        self.forward = torch.tensor([1.0, 1.0], device=self._device)
        self.backward = -self.forward

        return

    def set_action_and_observation_spaces(self) -> None:
        """
        Sets the action and observation spaces.
        """

        # Defines the observation space
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    np.ones(self._num_observations) * -np.Inf,
                    np.ones(self._num_observations) * np.Inf,
                ),
            }
        )

        # Defines the action space
        if self._discrete_actions == "MultiDiscrete":
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)] * self._max_actions)
        elif self._discrete_actions == "Continuous":
            self.action_space = spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32
            )
        elif self._discrete_actions == "Discrete":
            raise NotImplementedError("The Discrete control mode is not supported.")
        else:
            raise NotImplementedError(
                "The requested discrete action type is not supported."
            )

    def add_stats(self, names: List[str]) -> None:
        """
        Adds training statistics to be recorded during training.

        Args:
            names (List[str]): list of names of the statistics to be recorded.
        """

        for name in names:
            torch_zeros = lambda: torch.zeros(
                self._num_envs,
                dtype=torch.float,
                device=self._device,
                requires_grad=False,
            )
            if not name in self.episode_sums.keys():
                self.episode_sums[name] = torch_zeros()

    def cleanup(self) -> None:
        """
        Prepares torch buffers for RL data collection.
        """

        # prepare tensors
        self.obs_buf = {
            "state": torch.zeros(
                (self._num_envs, self._num_observations),
                device=self._device,
                dtype=torch.float,
            ),
        }

        self.states_buf = torch.zeros(
            (self._num_envs, self.num_states), device=self._device, dtype=torch.float
        )
        self.rew_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.float
        )
        self.reset_buf = torch.ones(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.progress_buf = torch.zeros(
            self._num_envs, device=self._device, dtype=torch.long
        )
        self.extras = {}

    def set_up_scene(self, scene) -> None:
        """
        Sets up the USD scene inside Omniverse for the task.

        Args:
            scene (Usd.Stage): the USD scene to be set up."""

        # Add the floating platform, and the marker
        self.get_heron()
        self.get_target()
        self.get_USV_dynamics()

        RLTask.set_up_scene(self, scene, replicate_physics=False)

        # Collects the interactive elements in the scene
        root_path = "/World/envs/.*/heron"
        self._heron = HeronView(prim_paths_expr=root_path, name="heron_view", track_contact_force=True)

        # Add views to scene
        scene.add(self._heron)
        scene.add(self._heron.base)

        scene.add(self._heron.thruster_left)
        scene.add(self._heron.thruster_right)

        # Add arrows to scene if task is go to pose
        scene, self._marker = self.task.add_visual_marker_to_scene(scene)
        return

    def get_heron(self):
        """
        Adds the floating platform to the scene."""

        asv = Heron(
            prim_path=self.default_zero_env_path + "/heron",
            name="heron",
            translation=self._asv_position,
        )
        self._sim_config.apply_articulation_settings(
            "heron",
            get_prim_at_path(asv.prim_path),
            self._sim_config.parse_actor_config("heron"),
        )

    def get_target(self) -> None:
        """
        Adds the visualization target to the scene."""

        self.task.generate_target(
            self.default_zero_env_path, self._default_marker_position
        )

    def get_USV_dynamics(self):
        """create physics"""
        self.hydrostatics = Hydrostatics(
            num_envs=self.num_envs,
            device=self._device,
            gravity=self.gravity,
            params=self.hydrostatics_cfg,
        )
        self.hydrodynamics = Hydrodynamics(
            dr_params=self._task_cfg["robot"]["asv_domain_randomization"]["drag"],
            num_envs=self.num_envs,
            device=self._device,
            params=self.hydrodynamics_cfg,
        )
        self.thrusters_dynamics = DynamicsFirstOrder(
            dr_params=self._task_cfg["robot"]["asv_domain_randomization"]["thruster"],
            num_envs=self.num_envs,
            device=self._device,
            timeConstant=self.timeConstant,
            dt=self.dt,
            params=self.thrusters_dynamics_cfg,
        )

    def update_state(self) -> None:
        """
        Updates the state of the system.
        """

        # Collects the position and orientation of the platform
        self.root_pos, self.root_quats = self._heron.get_world_poses(clone=True)
        # Remove the offset from the different environments
        root_positions = self.root_pos - self._env_pos
        # Collects the velocity of the platform
        self.root_velocities = self._heron.get_velocities(clone=True)
        root_velocities = self.root_velocities.clone()
        # Cast quaternion to Yaw
        siny_cosp = 2 * (
            self.root_quats[:, 0] * self.root_quats[:, 3]
            + self.root_quats[:, 1] * self.root_quats[:, 2]
        )
        cosy_cosp = 1 - 2 * (
            self.root_quats[:, 2] * self.root_quats[:, 2]
            + self.root_quats[:, 3] * self.root_quats[:, 3]
        )
        orient_z = torch.arctan2(siny_cosp, cosy_cosp)
        # Add noise on obs
        root_positions = self.DR.noisy_observations.add_noise_on_pos(
            root_positions, step=self.step
        )
        root_velocities = self.DR.noisy_observations.add_noise_on_vel(
            root_velocities, step=self.step
        )
        orient_z = self.DR.noisy_observations.add_noise_on_heading(
            orient_z, step=self.step
        )
        #net_contact_forces = self.compute_contact_forces()
        # Compute the heading
        self.heading[:, 0] = torch.cos(orient_z)
        self.heading[:, 1] = torch.sin(orient_z)

        net_contact_forces = self.compute_contact_forces()
        # Dump to state
        self.current_state = {
            "position": root_positions[:, :2],
            "orientation": self.heading,
            "linear_velocity": root_velocities[:, :2],
            "angular_velocity": root_velocities[:, -1],
            "net_contact_forces": net_contact_forces,
        }

    def compute_contact_forces(self) -> torch.Tensor:
        """
        Get the contact forces of the platform.

        Returns:
            net_contact_forces_norm (torch.Tensor): the norm of the net contact forces.
        """
        net_contact_forces = self._heron.base.get_net_contact_forces(clone=False)
        return torch.norm(net_contact_forces, dim=-1)


    def get_observations(self) -> Dict[str, torch.Tensor]:
        """
        Gets the observations of the task to be passed to the policy.

        Returns:
            observations: a dictionary containing the observations of the task.
        """

        # implement logic to retrieve observation states
        self.update_state()
        # Get the state
        self.obs_buf["state"] = self.task.get_state_observations(self.current_state)

        observations = {self._heron.name: {"obs_buf": self.obs_buf}}
        return observations

    def pre_physics_step(self, actions: torch.Tensor) -> None:
        """
        This function implements the logic to be performed before physics steps.

        Args:
            actions (torch.Tensor): the actions to be applied to the platform.
        """

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

        # Remap actions to the correct values
        if self._discrete_actions == "MultiDiscrete":
            # If actions are multidiscrete [0, 1]
            thrust_cmds = self.actions.float() * 2 - 1
        elif self._discrete_actions == "Continuous":
            # Transform continuous actions to [-1, 1] discrete actions.
            thrust_cmds = self.actions.float()
        else:
            raise NotImplementedError("")

        # Applies the thrust multiplier
        thrusts = thrust_cmds

        # Adds random noise on the actions
        # thrusts = self.DR.noisy_actions.add_noise_on_act(thrusts, step=self.step)

        # Clip the actions
        thrusts = torch.clamp(thrusts, -1.0, 1.0)

        # clear actions for reset envs
        thrusts[reset_env_ids] = 0

        self.thrusters_dynamics.set_target_force(thrusts)

        return

    def apply_forces(self) -> None:
        """
        Applies all the forces to the platform and its thrusters."""
        disturbance_forces = self.DR.force_disturbances.get_force_disturbance(self.root_pos)
        torque_disturbance = self.DR.torque_disturbances.get_torque_disturbance(self.root_pos)
        # Hydrostatic force
        self.hydrostatic_force[:, :] = (
            self.hydrostatics.compute_archimedes_metacentric_local(self.root_pos, self.root_quats)
        )
        # Hydrodynamic forces
        self.drag[:, :] = self.hydrodynamics.ComputeHydrodynamicsEffects(
            self.root_quats,
            self.root_velocities[:, :],
        )

        self.thrusters[:, :] = self.thrusters_dynamics.update_forces()

        self._heron.base.apply_forces_and_torques_at_pos(
            forces=
            disturbance_forces
            + self.hydrostatic_force[:, :3]
            + self.drag[:, :3],
            torques=
            torque_disturbance
            + self.hydrostatic_force[:, 3:]
            + self.drag[:, 3:],
            is_global=False,
        )

        self._heron.thruster_left.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, :3], is_global=False
        )
        self._heron.thruster_right.apply_forces_and_torques_at_pos(
            forces=self.thrusters[:, 3:], is_global=False
        )

    def post_reset(self):
        """
        This function implements the logic to be performed after a reset.
        """

        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._heron.get_world_poses()
        self.root_velocities = self._heron.get_velocities()
        self.dof_pos = self._heron.get_joint_positions()
        self.dof_vel = self._heron.get_joint_velocities()

        # Set initial conditions
        self.initial_root_pos, self.initial_root_rot = (
            self.root_pos.clone(),
            self.root_rot.clone(),
        )
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros(
            (self.num_envs, 4), dtype=torch.float32, device=self._device
        )
        self.initial_pin_rot[:, 0] = 1
        # Set the initial contact state
        self.contact_state = torch.zeros(
            (self._num_envs),
            dtype=torch.float32,
            device=self._device,
        )

        # control parameters
        self.thrusts = torch.zeros(
            (self._num_envs, self._max_actions, 3),
            dtype=torch.float32,
            device=self._device,
        )

        self.set_targets(self.all_indices)

    def set_targets(self, env_ids: torch.Tensor):
        """
        Sets the targets for the task.

        Args:
            env_ids (torch.Tensor): the indices of the environments for which to set the targets.
        """

        num_sets = len(env_ids)
        env_long = env_ids.long()     
        # Randomizes the position of the ball on the x y axes
        target_positions, target_orientation = self.task.get_goals(
            env_long,
            step=self.step,
        )
        if len(target_positions.shape) == 3:
            position = (
                target_positions
                + self.initial_pin_pos[env_long]
                .view(num_sets, 1, 3)
                .expand(*target_positions.shape)
            ).reshape(-1, 3)
            a = (env_long * target_positions.shape[1]).repeat_interleave(
                target_positions.shape[1]
            )
            b = torch.arange(target_positions.shape[1], device=self._device).repeat(
                target_positions.shape[0]
            )
            env_long = a + b
            target_orientation = target_orientation.reshape(-1, 4)
        else:
            position = target_positions + self.initial_pin_pos[env_long]

        ## Apply the new goals
        if self._marker:
            self._marker.set_world_poses(
                position,
                target_orientation,
                indices=env_long,
            )

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        """
        Resets the environments with the given indices.

        Args:
            env_ids (torch.Tensor): the indices of the environments to be reset."""

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        self.set_targets(env_ids)
        self.DR.force_disturbances.generate_forces(env_ids, num_resets, step=self.step)
        self.DR.torque_disturbances.generate_torques(env_ids, num_resets, step=self.step)

        # randomize masses, coms, and inertias
        self.DR.mass_disturbances.randomize_masses(env_ids, step=self.step)

        # Updates the masses of the platforms
        masses = self.DR.mass_disturbances.get_masses(env_ids)
        self._heron.base.set_masses(masses=masses, indices=env_ids)

        # Read the initial CoM position and orientation
        com_pos, com_ori = self._heron.base.get_coms(indices=env_ids)

        # Randomize the CoM position only in 2D
        com_pos_2d = self.DR.mass_disturbances.get_CoM(env_ids)
        com_pos[:, 0, :2] = com_pos_2d[:, :] # Map to 3d position

        # Updates the coms of the platforms
        self._heron.base.set_coms(com_pos, com_ori, indices=env_ids)

        # Read the initial moments of inertia
        inertias = self._heron.base.get_inertias(indices=env_ids, clone=True) # (num_envs, 9)
        # Randomize the moments of inertia only in the z axis
        mi_z = self.DR.mass_disturbances.get_moments_of_inertia(env_ids)
        inertias[:, -1] = mi_z[:] # Set the last element of the diagonal matrix (zz)
        # Updates the inertias of the platforms
        self._heron.base.set_inertias(values=inertias, indices=env_ids)
        inertias = self._heron.base.get_inertias(indices=env_ids, clone=True) # (num_envs, 9)

        # Resets hydrodynamic coefficients
        self.hydrodynamics.reset_coefficients(env_ids, num_resets)
        # Resets thruster randomization
        self.thrusters_dynamics.reset_thruster_randomization(env_ids, num_resets)
        # Randomizes the starting position of the platform within a disk around the target
        root_pos, root_quat, root_vel = self.task.get_initial_conditions(env_ids, step=self.step)
        root_pos[:, :2] = root_pos[:, :2] + self.initial_root_pos[env_ids, :2]
        root_pos[:, 2] = 0.1
        self._heron.set_world_poses(
            root_pos, root_quat, indices=env_ids
        )
        self._heron.set_velocities(root_vel, indices=env_ids)

        # Resets the states of the joints
        self.dof_pos[env_ids, :] = torch.zeros(
            (num_resets, self._heron.num_dof), device=self._device
        )
        self.dof_vel[env_ids, :] = 0
        self._heron.set_joint_positions(self.dof_pos[env_ids], indices=env_ids)

        # Sets the velocities to 0
        root_velocities = self.root_velocities.clone()
        self._heron.set_joint_velocities(self.dof_vel[env_ids], indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        
        # fill `extras`
        self.extras["episode"] = {}
        self.extras_wandb = {}
        for key in self.episode_sums.keys():
            value = (
                torch.mean(self.episode_sums[key][env_ids]) / self._max_episode_length
            )
            if key in self._penalties.get_stats_name():
                self.extras_wandb[key] = value
            elif key in self.task.log_with_wandb:
                self.extras_wandb[key] = value
            else:
                self.extras["episode"][key] = value
            self.episode_sums[key][env_ids] = 0.0

    def update_state_statistics(self) -> None:
        """
        Updates the statistics of the state of the training.
        """

        self.episode_sums["normed_linear_vel"] += torch.norm(
            self.current_state["linear_velocity"], dim=-1
        )
        self.episode_sums["normed_angular_vel"] += torch.abs(
            self.current_state["angular_velocity"]
        )
        self.episode_sums["actions_sum"] += torch.sum(self.actions, dim=-1)

    def calculate_metrics(self) -> None:
        """
        Calculates the metrics of the training.
        That is the rewards, penalties, and other perfomance statistics.
        """

        position_reward = self.task.compute_reward(self.current_state, self.actions)
        self.iteration += 1
        self.step += 1 / self._task_cfg["env"]["horizon_length"]
        penalties = self._penalties.compute_penalty(
            self.current_state, self.actions, self.step
        )
        self.rew_buf[:] = position_reward - penalties
        self.episode_sums = self.task.update_statistics(self.episode_sums)
        self.episode_sums = self._penalties.update_statistics(self.episode_sums)
        if self._enable_wandb_logs:
            if self.iteration / self._task_cfg["env"]["horizon_length"] % 1 == 0:
                self.extras_wandb["wandb_step"] = int(self.step)
                for key, value in self._penalties.get_logs().items():
                    self.extras_wandb[key] = value
                for key, value in self.task.get_logs(self.step).items():
                    self.extras_wandb[key] = value
                for key, value in self.DR.get_logs(self.step).items():
                    self.extras_wandb[key] = value
                wandb.log(self.extras_wandb)
                self.extras_wandb = {}

    def is_done(self) -> None:
        """
        Checks if the episode is done.
        """

        # resets due to misbehavior
        ones = torch.ones_like(self.reset_buf)
        die = self.task.update_kills()

        # resets due to episode length
        self.reset_buf[:] = torch.where(
            self.progress_buf >= self._max_episode_length - 1, ones, die
        )
