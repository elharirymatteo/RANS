__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.AGV_skidsteer_2W import (
    AGV_SkidSteer_2W,
)
from omniisaacgymenvs.robots.articulations.views.AGV_skidsteer_2W_view import (
    AGVSkidSteer2WView,
)
from omniisaacgymenvs.tasks.common_3DoF.task_factory import (
    task_factory,
)
from omniisaacgymenvs.tasks.common_3DoF.penalties import (
    EnvironmentPenalties,
)
from omniisaacgymenvs.tasks.common_3DoF.disturbances import (
    Disturbances,
)
from omniisaacgymenvs.robots.actuators.dynamics import Actuator

from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.torch.rotations import *
from typing import Dict, List, Tuple
from gym import spaces
import numpy as np
import wandb
import torch

EPS = 1e-6  # small constant to avoid divisions by 0 and log(0)


class AGVSS2WVirtual(RLTask):
    """
    The main class used to run tasks on the floating platform.
    Unlike other class in this repo, this class can be used to run different tasks.
    The idea being to extend it to multitask RL in the future.
    """

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
        self._agv_cfg = self._task_cfg["robot"]
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._max_episode_length = self._task_cfg["env"]["maxEpisodeLength"]
        self._discrete_actions = self._task_cfg["env"]["action_mode"]
        self._device = self._cfg["sim_device"]
        self.iteration = 0
        self.step = 0

        # Collects the platform parameters
        self.dt = self._task_cfg["sim"]["dt"]
        # Collects the task parameters
        task_cfg = self._task_cfg["sub_task"]
        reward_cfg = self._task_cfg["reward"]
        penalty_cfg = self._task_cfg["penalty"]
        domain_randomization_cfg = self._task_cfg["disturbances"]
        # Instantiate the task, reward and platform
        self.task = task_factory.get(task_cfg, reward_cfg, self._num_envs, self._device)
        self._penalties = EnvironmentPenalties(**penalty_cfg)
        self.DR = Disturbances(
            domain_randomization_cfg,
            num_envs=self._num_envs,
            device=self._device,
        )
        self._num_observations = self.task._num_observations
        self._max_actions = 2
        self._num_actions = 2
        RLTask.__init__(self, name, env)
        # Instantiate the action and observations spaces
        self.set_action_and_observation_spaces()
        # Sets the initial positions of the target and platform
        self._fp_position = torch.tensor([0, 0.0, 0.5])
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
        self.contact_state = torch.zeros(
            (self._num_envs), device=self._device, dtype=torch.float32
        )
        # Extra info
        self.extras = {}
        self.extras_wandb = {}
        # Episode statistics
        self.episode_sums = self.task.create_stats({})
        self.add_stats(self._penalties.get_stats_name())
        self.add_stats(["normed_linear_vel", "normed_angular_vel", "actions_sum"])
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
                "masks": spaces.Box(low=0, high=1, shape=(self._max_actions,)),
                "masses": spaces.Box(low=-np.inf, high=np.inf, shape=(3,)),
            }
        )

        # Defines the action space
        if self._discrete_actions == "MultiDiscrete":
            # RLGames implementation of MultiDiscrete action space requires a tuple of Discrete spaces
            self.action_space = spaces.Tuple([spaces.Discrete(2)] * self._max_actions)
        elif self._discrete_actions == "Continuous":
            pass
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
            "masks": torch.zeros(
                (self._num_envs, self._max_actions),
                device=self._device,
                dtype=torch.float,
            ),
            "masses": torch.zeros(
                (self._num_envs, 3),
                device=self._device,
                dtype=torch.float,
            ),
        }

        self.states_buf = torch.zeros(
            (self._num_envs, self._num_states), device=self._device, dtype=torch.float
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
        self.extras_wandb = {}

    def set_up_scene(self, scene) -> None:
        """
        Sets up the USD scene inside Omniverse for the task.

        Args:
            scene (Usd.Stage): the USD scene to be set up.
        """

        # Add the floating platform, and the marker
        self.get_agv_skidsteer()
        self.get_target()

        RLTask.set_up_scene(self, scene, replicate_physics=False)

        # Collects the interactive elements in the scene
        root_path = "/World/envs/.*/AGV_SS_2W"
        self._agvs = AGVSkidSteer2WView(
            prim_paths_expr=root_path,
            name="agv_skidsteer_2w_view",
            track_contact_force=True,
        )
        # Add views to scene
        scene.add(self._agvs)
        scene.add(self._agvs.base)

        # Add arrows to scene if task is go to pose
        scene, self._marker = self.task.add_visual_marker_to_scene(scene)
        return

    def get_agv_skidsteer(self):
        """
        Adds the floating platform to the scene.
        """

        agv_ss = AGV_SkidSteer_2W(
            prim_path=self.default_zero_env_path + "/AGV_SS_2W",
            name="agv_skidsteer_2w",
            translation=self._fp_position,
            cfg=self._agv_cfg,
        )
        self._sim_config.apply_articulation_settings(
            "agv_skidsteer_2w",
            get_prim_at_path(agv_ss.prim_path),
            self._sim_config.parse_actor_config("agv_skidsteer_2w"),
        )
        self.build_actuators_dynamics(agv_ss, self._num_actions)

    def build_actuators_dynamics(self, agv_ss: AGV_SkidSteer_2W, num_act) -> None:
        """
        Builds the dynamics of the actuators.
        """

        self.actuators_dynamics = [Actuator(self.dt, self._num_envs, self._device, agv_ss._settings.actuators)  for i in range(num_act)]

    def get_target(self) -> None:
        """
        Adds the visualization target to the scene.
        """

        self.task.generate_target(
            self.default_zero_env_path, self._default_marker_position
        )

    def update_state(self) -> None:
        """
        Updates the state of the system.
        """

        # Collects the position and orientation of the platform
        self.root_pos, self.root_quats = self._agvs.get_world_poses(
            clone=True
        )
        # Remove the offset from the different environments
        root_positions = self.root_pos - self._env_pos
        # Collects the velocity of the platform
        self.root_velocities = self._agvs.get_velocities(clone=True)
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
        net_contact_forces = self.compute_contact_forces()
        # Compute the heading
        self.heading[:, 0] = torch.cos(orient_z)
        self.heading[:, 1] = torch.sin(orient_z)

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
        net_contact_forces = self._agvs.base.get_net_contact_forces(clone=False)
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
        # Get the action masks
        #self.obs_buf["masks"] = self.virtual_platform.action_masks
        self.obs_buf["masses"] = self.DR.mass_disturbances.get_masses_and_com()

        observations = {self._agvs.name: {"obs_buf": self.obs_buf}}
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
            velocity_cmds = self.actions.float() / 2
        elif self._discrete_actions == "Continuous":
            velocity_cmds = torch.clamp(self.actions, min=-0.5, max=0.5)
        else:
            raise NotImplementedError("")
        # Applies the scaling and dynamics to the actions expects actions in the -0.5, 0.5 range.
        velocity_cmds[:, 0] = self.actuators_dynamics[0].apply_dynamics_torch(velocity_cmds[:, 0])
        velocity_cmds[:, 1] = self.actuators_dynamics[1].apply_dynamics_torch(velocity_cmds[:, 1])
        # Adds random noise on the actions
        velocity_cmds = self.DR.noisy_actions.add_noise_on_act(velocity_cmds, step=self.step)
        # clear actions for reset envs
        velocity_cmds[reset_env_ids] = 0
        self._agvs.set_joint_velocities(velocity_cmds)
        return

    def apply_forces(self) -> None:
        """
        Applies all the forces to the platform and its thrusters.
        """

        # Applies the domain randomization
        floor_forces = self.DR.force_disturbances.get_force_disturbance(self.root_pos)
        torque_disturbance = self.DR.torque_disturbances.get_torque_disturbance(
            self.root_pos
        )
        self._agvs.base.apply_forces_and_torques_at_pos(
            forces=floor_forces,
            torques=torque_disturbance,
            positions=self.root_pos,
            is_global=True,
        )

    def post_reset(self):
        """
        This function implements the logic to be performed after a reset.
        """

        # implement any logic required for simulation on-start here
        self.root_pos, self.root_rot = self._agvs.base.get_world_poses()
        self.root_velocities = self._agvs.base.get_velocities()
        self.dof_pos = self._agvs.get_joint_positions()
        self.dof_vel = self._agvs.get_joint_velocities()

        # Set initial conditions
        self.initial_root_pos, self.initial_root_rot = (
            self.root_pos.clone(),
            self.root_rot.clone(),
        )
        self.initial_pin_pos = self._env_pos
        self.initial_pin_rot = torch.zeros(
            (self._num_envs, 4), dtype=torch.float32, device=self._device
        )
        self.initial_pin_rot[:, 0] = 1
        # Set the initial contact state
        self.contact_state = torch.zeros(
            (self._num_envs),
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
            env_ids (torch.Tensor): the indices of the environments to be reset.
        """

        num_resets = len(env_ids)
        # Resets the counter of steps for which the goal was reached
        self.task.reset(env_ids)
        self.set_targets(env_ids)
        self.DR.force_disturbances.generate_forces(env_ids, num_resets, step=self.step)
        self.DR.torque_disturbances.generate_torques(
            env_ids, num_resets, step=self.step
        )
        self.DR.mass_disturbances.randomize_masses(env_ids, step=self.step)
        random_mass = self.DR.mass_disturbances.get_masses(env_ids)
        # Resets the actuators states
        for actuator in self.actuators_dynamics:
            actuator.reset_torch(env_ids)
        # Randomizes the starting position of the platform
        pos, quat, vel = self.task.get_initial_conditions(env_ids, step=self.step)
        siny_cosp = 2 * quat[:, 0] * quat[:, 3]
        cosy_cosp = 1 - 2 * (quat[:, 3] * quat[:, 3])
        h = torch.arctan2(siny_cosp, cosy_cosp)
        # Resets the contacts
        self.contact_state[env_ids] = 0
        # self._platforms.CoM.set_masses(random_mass, indices=env_ids)

        # apply resets
        self._agvs.set_world_poses(pos + self._env_pos[env_ids], quat, indices=env_ids)
        self._agvs.set_velocities(vel, indices=env_ids)
        dof_pos = torch.zeros(
            (num_resets, self._agvs.num_dof), device=self._device
        )
        self._agvs.set_joint_positions(dof_pos, indices=env_ids)

        dof_vel = torch.zeros(
            (num_resets, self._agvs.num_dof), device=self._device
        )
        self._agvs.set_joint_velocities(dof_vel, indices=env_ids)

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

        self.update_state_statistics()

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
