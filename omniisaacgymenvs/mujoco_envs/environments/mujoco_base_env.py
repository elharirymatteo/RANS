__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from typing import Dict, Union, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mujoco
import math
import os

from omniisaacgymenvs.mujoco_envs.environments.disturbances import (
    NoisyActions,
    NoisyObservations,
    TorqueDisturbance,
    UnevenFloorDisturbance,
    RandomKillThrusters,
    RandomSpawn,
)


def parseEnvironmentConfig(
    cfg: Dict[str, Union[float, int, Dict]]
) -> Dict[str, Union[float, int, Dict]]:
    """
    Parses the environment configuration from the config file.

    Args:
        cfg (Dict[str, Union[float, int, Dict]]): The configuration dictionary.

    Returns:
        Dict[str, Union[float, int, Dict]]: The parsed configuration dictionary."""

    new_cfg = {}
    new_cfg["disturbances"] = {}
    new_cfg["disturbances"]["seed"] = cfg["seed"]
    new_cfg["disturbances"]["use_uneven_floor"] = cfg["task"]["env"]["use_uneven_floor"]
    new_cfg["disturbances"]["use_sinusoidal_floor"] = cfg["task"]["env"][
        "use_sinusoidal_floor"
    ]
    new_cfg["disturbances"]["floor_min_freq"] = cfg["task"]["env"]["floor_min_freq"]
    new_cfg["disturbances"]["floor_max_freq"] = cfg["task"]["env"]["floor_max_freq"]
    new_cfg["disturbances"]["floor_min_offset"] = cfg["task"]["env"]["floor_min_offset"]
    new_cfg["disturbances"]["floor_max_offset"] = cfg["task"]["env"]["floor_max_offset"]
    new_cfg["disturbances"]["min_floor_force"] = cfg["task"]["env"]["min_floor_force"]
    new_cfg["disturbances"]["max_floor_force"] = cfg["task"]["env"]["max_floor_force"]

    new_cfg["disturbances"]["use_torque_disturbance"] = cfg["task"]["env"][
        "use_torque_disturbance"
    ]
    new_cfg["disturbances"]["use_sinusoidal_torque"] = cfg["task"]["env"][
        "use_sinusoidal_torque"
    ]
    new_cfg["disturbances"]["min_torque"] = cfg["task"]["env"]["min_torque"]
    new_cfg["disturbances"]["max_torque"] = cfg["task"]["env"]["max_torque"]

    new_cfg["disturbances"]["add_noise_on_pos"] = cfg["task"]["env"]["add_noise_on_pos"]
    new_cfg["disturbances"]["position_noise_min"] = cfg["task"]["env"][
        "position_noise_min"
    ]
    new_cfg["disturbances"]["position_noise_max"] = cfg["task"]["env"][
        "position_noise_max"
    ]
    new_cfg["disturbances"]["add_noise_on_vel"] = cfg["task"]["env"]["add_noise_on_vel"]
    new_cfg["disturbances"]["velocity_noise_min"] = cfg["task"]["env"][
        "velocity_noise_min"
    ]
    new_cfg["disturbances"]["velocity_noise_max"] = cfg["task"]["env"][
        "velocity_noise_max"
    ]
    new_cfg["disturbances"]["add_noise_on_heading"] = cfg["task"]["env"][
        "add_noise_on_heading"
    ]
    new_cfg["disturbances"]["heading_noise_min"] = cfg["task"]["env"][
        "heading_noise_min"
    ]
    new_cfg["disturbances"]["heading_noise_max"] = cfg["task"]["env"][
        "heading_noise_max"
    ]

    new_cfg["disturbances"]["add_noise_on_act"] = cfg["task"]["env"]["add_noise_on_act"]
    new_cfg["disturbances"]["min_action_noise"] = cfg["task"]["env"]["min_action_noise"]
    new_cfg["disturbances"]["max_action_noise"] = cfg["task"]["env"]["max_action_noise"]

    new_cfg["spawn_parameters"] = {}
    new_cfg["spawn_parameters"]["seed"] = cfg["seed"]
    try:
        new_cfg["spawn_parameters"]["max_spawn_dist"] = cfg["task"]["env"][
            "task_parameters"
        ]["max_spawn_dist"]
    except:
        new_cfg["spawn_parameters"]["max_spawn_dist"] = 0
    try:
        new_cfg["spawn_parameters"]["min_spawn_dist"] = cfg["task"]["env"][
            "task_parameters"
        ]["min_spawn_dist"]
    except:
        new_cfg["spawn_parameters"]["min_spawn_dist"] = 0

    new_cfg["spawn_parameters"]["kill_dist"] = cfg["task"]["env"]["task_parameters"][
        "kill_dist"
    ]

    new_cfg["step_time"] = cfg["task"]["sim"]["dt"]
    new_cfg["duration"] = (
        cfg["task"]["env"]["maxEpisodeLength"] * cfg["task"]["sim"]["dt"]
    )
    new_cfg["inv_play_rate"] = cfg["task"]["env"]["controlFrequencyInv"]
    new_cfg["platform"] = cfg["task"]["env"]["platform"]
    new_cfg["platform"]["seed"] = cfg["seed"]
    return new_cfg


class MuJoCoFloatingPlatform:
    """
    A class for the MuJoCo Floating Platform environment."""

    def __init__(
        self,
        step_time: float = 0.02,
        duration: float = 60.0,
        inv_play_rate: int = 10,
        spawn_parameters: Dict[str, float] = None,
        platform: Dict[str, Union[bool, dict, float, str, int]] = None,
        disturbances: Dict[str, Union[bool, float]] = None,
        **kwargs
    ) -> None:
        """
        Initializes the MuJoCo Floating Platform environment.

        Args:
            step_time (float, optional): The time between steps in the simulation (seconds). Defaults to 0.02.
            duration (float, optional): The duration of the simulation (seconds). Defaults to 60.0.
            inv_play_rate (int, optional): The inverse of the play rate. Defaults to 10.
            spawn_parameters (Dict[str, float], optional): A dictionary containing the spawn parameters. Defaults to None.
            platform (Dict[str, Union[bool,dict,float,str,int]], optional): A dictionary containing the platform parameters. Defaults to None.
            disturbances (Dict[str, Union[bool, float]], optional): A dictionary containing the disturbances parameters. Defaults to None.
            **kwargs: Additional arguments."""

        self.inv_play_rate = inv_play_rate
        self.platform = platform

        self.AN = NoisyActions(disturbances)
        self.ON = NoisyObservations(disturbances)
        self.TD = TorqueDisturbance(disturbances)
        self.UF = UnevenFloorDisturbance(disturbances)

        self.TK = RandomKillThrusters(
            {
                "num_thrusters_to_kill": platform["randomization"]["max_thruster_kill"]
                * platform["randomization"]["kill_thrusters"],
                "seed": platform["seed"],
            }
        )
        self.RS = RandomSpawn(spawn_parameters)

        self.createModel()
        self.initializeModel()
        self.setupPhysics(step_time, duration)
        self.initForceAnchors()

        self.reset()
        self.csv_datas = []

    def reset(
        self,
        initial_position: List[float] = [0, 0, 0],
        initial_orientation: List[float] = [1, 0, 0, 0],
    ) -> None:
        """
        Resets the simulation.

        Args:
            initial_position (list, optional): The initial position of the body. Defaults to [0,0,0].
            initial_orientation (list, optional): The initial orientation of the body. Defaults to [1,0,0,0].
        """

        self.resetPosition(
            initial_position=initial_position, initial_orientation=initial_orientation
        )
        self.UF.generate_floor()
        self.TD.generate_torque()
        self.TK.generate_thruster_kills()

    def initializeModel(self) -> None:
        """
        Initializes the mujoco model for the simulation."""

        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "top")

    def setupPhysics(self, step_time: float, duration: float) -> None:
        """
        Sets up the physics parameters for the simulation.

        Args:
            step_time (float): The time between steps in the simulation (seconds).
            duration (float): The duration of the simulation (seconds)."""

        self.model.opt.timestep = step_time
        self.model.opt.gravity = [0, 0, 0]
        self.duration = duration

    def createModel(self) -> None:
        """
        A YAML style string that defines the MuJoCo model for the simulation.
        The mass is set to 5.32 kg, the radius is set to 0.31 m.
        The initial position is set to (3, 3, 0.4) m."""

        self.radius = self.platform["core"]["radius"]
        self.mass = self.platform["core"]["mass"]

        sphere_p1 = """
        <mujoco model="tippe top">
          <option integrator="RK4"/>
        
          <asset>
            <texture name="grid" type="2d" builtin="checker" rgb1=".1 .2 .3"
             rgb2=".2 .3 .4" width="300" height="300"/>
            <material name="grid" texture="grid" texrepeat="8 8" reflectance=".2"/>
          </asset>
        
          <worldbody>
            <geom size="10.0 10.0 .01" type="plane" material="grid"/>
            <light pos="0 0 10.0"/>
            <camera name="closeup" pos="0 -3 2" xyaxes="1 0 0 0 1 2"/>
            <body name="top" pos="0 0 .4">
              <freejoint/>
        """
        sphere_p2 = (
            '<geom name="ball" type="sphere" size="'
            + str(self.radius)
            + '" mass="'
            + str(self.mass)
            + '"/>'
        )
        sphere_p3 = """
            </body>
          </worldbody>
          <keyframe>
            <key name="idle" qpos="3 3 0.4 1 0 0 0" qvel="0 0 0 0 0 0" />
          </keyframe>
        </mujoco>
        """
        sphere = "\n".join([sphere_p1, sphere_p2, sphere_p3])

        self.model = mujoco.MjModel.from_xml_string(sphere)

    def initForceAnchors(self) -> None:
        """ "
        Defines where the forces are applied relatively to the center of mass of the body.
        self.forces: 8x3 array of forces, indicating the direction of the force.
        self.positions: 8x3 array of positions, indicating the position of the force."""

        self.max_thrust = self.platform["configuration"]["thrust_force"]

        self.forces = np.array(
            [
                [1, -1, 0],
                [-1, 1, 0],
                [1, 1, 0],
                [-1, -1, 0],
                [-1, 1, 0],
                [1, -1, 0],
                [-1, -1, 0],
                [1, 1, 0],
            ]
        )
        # Normalize the forces.
        self.forces = self.forces / np.linalg.norm(self.forces, axis=1).reshape(-1, 1)
        # Multiply by the max thrust.
        self.forces = self.forces * self.max_thrust

        self.positions = (
            np.array(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [-1, 1, 0],
                    [-1, 1, 0],
                    [-1, -1, 0],
                    [-1, -1, 0],
                    [1, -1, 0],
                    [1, -1, 0],
                ]
            )
            * 0.2192
        )

    def resetPosition(
        self,
        initial_position: List[float] = [0, 0],
        initial_orientation: List[float] = [1, 0, 0, 0],
    ) -> None:
        """
        Resets the position of the body and sets its velocity to 0.
        Resets the timer as well.

        Args:
            initial_position (list, optional): The initial position of the body. Defaults to [0,0].
            initial_orientation (list, optional): The initial orientation of the body. Defaults to [1,0,0,0].
        """

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:2] = initial_position[:2]
        self.data.qpos[3:7] = initial_orientation
        self.data.qvel = 0

    def applyForces(self, action: np.ndarray) -> None:
        """
        Applies the forces to the body.

        Args:
            action (np.ndarray): The actions to apply to the body."""

        self.data.qfrc_applied[...] = 0  # Clear applied forces.
        rmat = self.data.xmat[self.body_id].reshape(3, 3)  # Rotation matrix.
        p = self.data.xpos[self.body_id]  # Position of the body.

        # Compute the number of thrusters fired, split the pressure between the nozzles.
        factor = max(np.sum(action), 1)
        # For each thruster, apply a force if needed.
        for i in range(8):
            if (
                self.TK.killed_thrusters_id is not None
                and i in self.TK.killed_thrusters_id
            ):
                continue
            # The force applied is the action value (1 or 0), divided by the number of thrusters fired (factor),
            force = self.AN.add_noise_on_act(action[i])
            force = force * (1.0 / factor) * self.forces[i]
            # If the force is not zero, apply the force.
            if np.sum(np.abs(force)) > 0:
                force = np.matmul(rmat, force)  # Rotate the force to the global frame.
                p2 = (
                    np.matmul(rmat, self.positions[i]) + p
                )  # Compute the position of the force.
                mujoco.mj_applyFT(
                    self.model,
                    self.data,
                    force,
                    [0, 0, 0],
                    p2,
                    self.body_id,
                    self.data.qfrc_applied,
                )  # Apply the force.

        uf_forces = self.UF.get_floor_forces(self.data.qpos[:2])
        td_forces = self.TD.get_torque_disturbance(self.data.qpos[:2])
        mujoco.mj_applyFT(
            self.model,
            self.data,
            uf_forces,
            td_forces,
            self.data.qpos[:3],
            self.body_id,
            self.data.qfrc_applied,
        )  # Apply the force.

    def getObs(self) -> Dict[str, np.ndarray]:
        """
        returns an up to date observation buffer.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the state of the simulation.
        """

        state = {}
        state["angular_velocity"] = self.ON.add_noise_on_vel(self.data.qvel[3:6].copy())
        state["linear_velocity"] = self.ON.add_noise_on_vel(self.data.qvel[0:3].copy())
        state["position"] = self.ON.add_noise_on_pos(self.data.qpos[0:3].copy())
        state["quaternion"] = self.data.qpos[3:].copy()
        return state

    def runLoop(
        self,
        model,
        initial_position: List[float] = [0, 0],
        initial_orientation: List[float] = [1, 0, 0, 0],
    ):
        """
        Runs the simulation loop.

        Args:
            model (object): The model of the controller.
            initial_position (list, optional): The initial position of the body. Defaults to [0,0].
            initial_orientation (list, optional): The initial orientation of the body. Defaults to [1,0,0,0].
        """

        self.reset(
            initial_position=initial_position, initial_orientation=initial_orientation
        )

        done = False
        while (self.duration > self.data.time) and (not done):
            state = self.getObs()  # Updates the state of the simulation.
            # Get the actions from the controller
            self.actions = model.getAction(state)
            # Plays only once every self.inv_play_rate steps.
            for _ in range(self.inv_play_rate):
                self.applyForces(self.actions)
                mujoco.mj_step(self.model, self.data)
            done = model.isDone()
