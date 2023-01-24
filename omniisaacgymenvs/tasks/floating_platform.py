
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.floating_platform import FloatingPlatform
from omniisaacgymenvs.robots.articulations.views.floating_platform_view import FloatingPlarformView
from omni.isaac.core.articulations import ArticulationView

from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.objects import DynamicSphere
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path

import numpy as np
import torch

EPS = 1e-6   # small constant to avoid divisions by 0 and log(0)



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

        self._num_observations = 4
        self._num_actions = 1

        self._fp_position = torch.tensor([0, 0, 1.0])
        self._ball_position = torch.tensor([0, 0, 1.0])
        # call parent class’s __init__
        RLTask.__init__(self, name, env)

def set_up_scene(self, scene) -> None:
    fp = self.get_floating_platform()
    # self.get_target()
    # implement environment setup here
    scene.add(fp) # add a robot actor to the stage
    RLTask.set_up_scene(self, scene) # pass scene to parent class - this method in RLTask also uses GridCloner to clone the robot and adds a ground plane if desired

    self._my_robots = ArticulationView(prim_paths_expr=f"/robot", name=f"floating_platform") # create a view of robots
    self._balls = RigidPrimView(prim_paths_expr="/World/envs/.*/ball")

    scene.add(self._my_robots) # add view to scene for initialization
    scene.add(self._balls)
    return

def get_floating_platform(self):
    fp = FloatingPlatform(prim_path=self.default_zero_env_path + "/robot", name="floating_platform",
                        translation=self._fp_position)
    return fp
    # self._sim_config.apply_articulation_settings("floating_platform", get_prim_at_path(fp.prim_path),
    #                                                self._sim_config.parse_actor_config("floating_platofrm"))

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

def post_reset(self):
    # implement any logic required for simulation on-start here
    pass

def pre_physics_step(self, actions: torch.Tensor) -> None:
    # implement logic to be performed before physics steps
    self.perform_reset()
    self.apply_action(actions)

def get_observations(self) -> dict:
    # implement logic to retrieve observation states
    self.root_pos, self.root_rot = self._my_robots.get_world_poses(clone=False)
    self.root_velocities = self._my_robots.get_velocities(clone=False)

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
        self._my_robots.name: {
            "obs_buf": self.obs_buf
        }
    }
    return observations


def calculate_metrics(self) -> None:
    # implement logic to compute rewards
    self.rew_buf = self.compute_rewards()

def is_done(self) -> None:
    # implement logic to update dones/reset buffer
    self.reset_buf = self.compute_resets()