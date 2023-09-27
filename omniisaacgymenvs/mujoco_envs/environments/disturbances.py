import numpy as np
import math

class RandomSpawn:
    def __init__(self, cfg):
        self._rng = np.random.default_rng(seed=cfg['seed'])
        self._max_spawn_dist = cfg["max_spawn_dist"]
        self._min_spawn_dist = cfg["min_spawn_dist"]
        self._kill_dist = cfg["kill_dist"]

    def getInitialCondition(self):
        theta = self._rng.uniform(-np.pi, np.pi, 1)
        r = self._rng.uniform(self._min_spawn_dist, self._max_spawn_dist)
        initial_position = [np.cos(theta) * r, np.sin(theta) * r]
        heading = self._rng.uniform(-np.pi, np.pi, 1)
        initial_orientation = [np.cos(heading*0.5), 0, 0, np.sin(heading*0.5)]
        return initial_position, initial_orientation


class RandomKillThrusters:
    """
    Randomly kills thrusters."""

    def __init__(self, cfg):
        self._rng = np.random.default_rng(seed=cfg['seed'])
        self._num_thrusters_to_kill = cfg['num_thrusters_to_kill']
        self.killed_thrusters_id = []

    def generate_thruster_kills(self):
        self.killed_thrusters_id = self._rng.choice(8, self._num_thrusters_to_kill, replace=False) #[2,3]
        print("Killed thrusters: ", self.killed_thrusters_id)


class UnevenFloorDisturbance:
    """
    Creates disturbances on the platform by simulating an uneven floor."""

    def __init__(self, cfg: dict) -> None:
        # Uneven floor generation
        self._rng = np.random.default_rng(seed=cfg["seed"])
        self._use_uneven_floor = cfg['use_uneven_floor']
        self._use_sinusoidal_floor = cfg['use_sinusoidal_floor']
        self._min_freq = cfg['floor_min_freq']
        self._max_freq = cfg['floor_max_freq']
        self._min_offset = cfg['floor_min_offset']
        self._max_offset = cfg['floor_max_offset']
        self._max_floor_force = cfg['max_floor_force'] 
        self._min_floor_force = cfg['min_floor_force'] 
        self._max_floor_force = math.sqrt(self._max_floor_force**2 / 2)
        self._min_floor_force = math.sqrt(self._min_floor_force**2 / 2)

        self._floor_forces = np.zeros(3, dtype=np.float32)
        self._floor_x_freq = 0
        self._floor_y_freq = 0
        self._floor_x_offset = 0
        self._floor_y_offset = 0

    def generate_floor(self) -> None:
        """
        Generates the uneven floor."""
        if self._use_uneven_floor:
            if self._use_sinusoidal_floor:
                self._floor_x_freq   = self._rng.uniform(self._min_freq, self._max_freq, 1)
                self._floor_y_freq   = self._rng.uniform(self._min_freq, self._max_freq, 1)
                self._floor_x_offset = self._rng.uniform(self._min_offset, self._max_offset, 1)
                self._floor_y_offset = self._rng.uniform(self._min_offset, self._max_offset, 1)
            else:
                r = self._rng.uniform(self._min_floor_force, self._max_floor_force, 1)
                theta = self._rng.uniform(0, 1, 1) * math.pi * 2
                self._floor_forces[0] = np.cos(theta) * r
                self._floor_forces[1] = np.sin(theta) * r

    def get_floor_forces(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the floor forces for the current state of the robot."""
        if self._use_uneven_floor:
            if self._use_sinusoidal_floor:
                self._floor_forces[0] = np.sin(root_pos[0] * self._floor_x_freq + self._floor_x_offset) * self._max_floor_force
                self._floor_forces[1] = np.sin(root_pos[1] * self._floor_y_freq + self._floor_y_offset) * self._max_floor_force
       
        return self._floor_forces


class TorqueDisturbance:
    """
    Creates disturbances on the platform by simulating a torque applied to its center."""

    def __init__(self, cfg: dict) -> None:
        self._rng = np.random.default_rng(seed=cfg["seed"])
        # Uneven floor generation
        self._use_torque_disturbance = cfg['use_torque_disturbance']
        self._use_sinusoidal_torque = cfg['use_sinusoidal_torque']
        self._max_torque = cfg['max_torque']
        self._min_torque = cfg['min_torque']

        # use the same min/max frequencies and offsets for the floor
        self._min_freq = cfg['floor_min_freq']
        self._max_freq = cfg['floor_max_freq']
        self._min_offset = cfg['floor_min_offset']
        self._max_offset = cfg['floor_max_offset']

        self._torque_forces = np.zeros(3, dtype=np.float32)
        self._torque_freq = 0
        self._torque_offset = 0

    def generate_torque(self) -> None:
        """
        Generates the torque disturbance."""
        if self._use_torque_disturbance:
            if self._use_sinusoidal_torque:
                #  use the same min/max frequencies and offsets for the floor
                self._torque_freq = self._rng.uniform(self._min_freq, self._max_freq, 1)
                self._torque_offset = self._rng.uniform(self._min_offset, self._max_offset, 1)
            else:
                r = self._rng.uniform(self._min_torque, self._max_torque, 1) * self._rng.choice([1,-1])
                self._torque_forces[2] = r

    def get_torque_disturbance(self, root_pos: np.ndarray) -> np.ndarray:
        """
        Computes the torque for the current state of the robot."""
        if self._use_torque_disturbance:
            if self._use_sinusoidal_torque:
                self._torque_forces[2] = np.sin(root_pos * self._torque_freq + self._torque_offset) * self._max_torque

        return self._torque_forces


class NoisyObservations:
    """
    Adds noise to the observations of the robot."""

    def __init__(self, cfg: dict) -> None:
        self._rng = np.random.default_rng(seed=cfg['seed'])
        self._add_noise_on_pos = cfg['add_noise_on_pos']
        self._position_noise_min = cfg['position_noise_min']
        self._position_noise_max = cfg['position_noise_max']
        self._add_noise_on_vel = cfg['add_noise_on_vel']
        self._velocity_noise_min = cfg['velocity_noise_min']
        self._velocity_noise_max = cfg['velocity_noise_max']
        self._add_noise_on_heading = cfg['add_noise_on_heading']
        self._heading_noise_min = cfg['heading_noise_min']
        self._heading_noise_max = cfg['heading_noise_max']
    
    def add_noise_on_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Adds noise to the position of the robot."""

        if self._add_noise_on_pos:
            pos += self._rng.uniform(self._position_noise_min, self._position_noise_max, pos.shape)
        return pos
    
    def add_noise_on_vel(self, vel: np.ndarray) -> np.ndarray:
        """
        Adds noise to the velocity of the robot."""

        if self._add_noise_on_vel:
            vel += self._rng.uniform(self._velocity_noise_min, self._velocity_noise_max, vel.shape)
        return vel
    
    def add_noise_on_heading(self, heading: np.ndarray) -> np.ndarray:
        """
        Adds noise to the heading of the robot."""

        if self._add_noise_on_heading:
            heading += self._rng.uniform(self._heading_noise_min, self._heading_noise_max, heading.shape)
        return heading


class NoisyActions:
    """
    Adds noise to the actions of the robot."""

    def __init__(self, cfg: dict) -> None:
        self._rng = np.random.default_rng(seed=cfg['seed'])
        self._add_noise_on_act = cfg['add_noise_on_act']
        self._min_action_noise = cfg['min_action_noise']
        self._max_action_noise = cfg['max_action_noise']

    def add_noise_on_act(self, act: np.ndarray) -> np.ndarray:
        """
        Adds noise to the actions of the robot."""

        if self._add_noise_on_act:
            act += self._rng.uniform(self._min_action_noise, self._max_action_noise, 1)
        return act
    

