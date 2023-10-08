__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.virtual_floating_platform.MFP2D_core import parse_data_dict
from dataclasses import dataclass
import torch
import math


@dataclass
class ConfigurationParameters:
    """
    Thruster configuration parameters."""

    use_four_configurations: bool = False
    num_anchors: int = 4
    offset: float = math.pi / 4
    thrust_force: float = 1.0
    visualize: bool = False
    save_path: str = "thruster_configuration.png"

    def __post_init__(self):
        assert self.num_anchors > 1, "num_anchors must be larger or equal to 2."


@dataclass
class PlatformParameters:
    """
    Platform physical parameters."""

    mass: float = 5.0
    radius: float = 0.25
    refinement: int = 2
    CoM: tuple = (0, 0, 0)
    shape: str = "sphere"


@dataclass
class PlatformRandomization:
    """
    Platform randomization parameters."""

    random_permutation: bool = False
    random_offset: bool = False
    randomize_thruster_position: bool = False
    min_random_radius: float = 0.125
    max_random_radius: float = 0.25
    random_theta: float = 0.125
    randomize_thrust_force: bool = False
    min_thrust_force: float = 0.5
    max_thrust_force: float = 1.0
    kill_thrusters: bool = False
    max_thruster_kill: int = 1


def compute_actions(cfg_param: ConfigurationParameters):
    """
    Computes the number of actions for the thruster configuration."""

    if cfg_param.use_four_configurations:
        return 10
    else:
        return cfg_param.num_anchors * 2


class VirtualPlatform:
    """
    Generates a virtual floating platform with thrusters."""

    def __init__(self, num_envs: int, platform_cfg: dict, device: str) -> None:
        self._num_envs = num_envs
        self._device = device

        # Generates dataclasses from the configuration file
        self.core_cfg = parse_data_dict(PlatformParameters(), platform_cfg["core"])
        self.rand_cfg = parse_data_dict(
            PlatformRandomization(), platform_cfg["randomization"]
        )
        self.thruster_cfg = parse_data_dict(
            ConfigurationParameters(), platform_cfg["configuration"]
        )
        # Computes the number of actions
        self._max_thrusters = compute_actions(self.thruster_cfg)
        # Sets the empty buffers
        self.transforms2D = torch.zeros(
            (num_envs, self._max_thrusters, 3, 3),
            device=self._device,
            dtype=torch.float32,
        )
        self.current_transforms = torch.zeros(
            (num_envs, self._max_thrusters, 5), device=self._device, dtype=torch.float32
        )
        self.action_masks = torch.zeros(
            (num_envs, self._max_thrusters), device=self._device, dtype=torch.long
        )
        self.thrust_force = torch.zeros(
            (num_envs, self._max_thrusters), device=self._device, dtype=torch.float32
        )
        # Creates a unit vector to project the forces
        self.create_unit_vector()

        # Generates a visualization file for the provided thruster configuration
        if self.thruster_cfg.visualize:
            self.generate_base_platforms(self._num_envs, torch.arange(self._num_envs))
            self.visualize(self.thruster_cfg.save_path)

    def create_unit_vector(self) -> None:
        """
        Creates a unit vector to project the forces.
        The forces are in 2D so the unit vector is a 2D vector."""

        tmp_x = torch.ones(
            (self._num_envs, self._max_thrusters, 1),
            device=self._device,
            dtype=torch.float32,
        )
        tmp_y = torch.zeros(
            (self._num_envs, self._max_thrusters, 1),
            device=self._device,
            dtype=torch.float32,
        )
        self.unit_vector = torch.cat([tmp_x, tmp_y], dim=-1)

    def project_forces(self, forces: torch.Tensor) -> list:
        """
        Projects the forces on the platform."""

        # Applies force scaling, applies action masking
        rand_forces = forces * self.thrust_force * (1 - self.action_masks)
        # Split transforms into translation and rotation
        R = self.transforms2D[:, :, :2, :2].reshape(-1, 2, 2)
        T = self.transforms2D[:, :, 2, :2].reshape(-1, 2)
        # Create a zero tensor to add 3rd dimmension
        zero = torch.zeros((T.shape[0], 1), device=self._device, dtype=torch.float32)
        # Generate positions
        positions = torch.cat([T, zero], dim=-1)
        # Project forces
        force_vector = self.unit_vector * rand_forces.view(-1, self._max_thrusters, 1)
        rotated_forces = torch.matmul(R.reshape(-1, 2, 2), force_vector.view(-1, 2, 1))
        projected_forces = torch.cat([rotated_forces[:, :, 0], zero], dim=-1)

        return positions, projected_forces

    def randomize_thruster_state(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Randomizes the spatial configuration of the thruster."""

        self.generate_base_platforms(num_resets, env_ids)

    def generate_base_platforms(self, num_envs: int, env_ids: torch.Tensor) -> None:
        """
        Generates the spatial configuration of the thruster."""

        # ====================
        # Basic thruster positioning
        # ====================

        # Generates a fixed offset between the heading and the first generated thruster
        random_offset = (
            torch.ones((self._num_envs), device=self._device)
            .view(-1, 1)
            .expand(self._num_envs, self._max_thrusters)
            * math.pi
            / self.thruster_cfg.num_anchors
        )
        # Adds a random offset to each simulated platform between the heading and the first generated thruster
        if self.rand_cfg.random_offset:
            random_offset += (
                torch.rand((self._num_envs), device=self._device)
                .view(-1, 1)
                .expand(self._num_envs, self._max_thrusters)
                * math.pi
                * 2
            )
        # Generates a 180 degrees offset between two consecutive thruster (+/- 90 degrees).
        thrust_90 = (
            (
                torch.arange(2, device=self._device)
                .repeat(self._max_thrusters // 2)
                .expand(self._num_envs, self._max_thrusters)
                * 2
                - 1
            )
            * math.pi
            / 2
        )
        # If four configurations, it generates platforms with 4, 6, 8, and 10 thrusters.
        if self.thruster_cfg.use_four_configurations:
            # Generates N, two by two thruster
            thrust_offset1 = (
                torch.arange(5, device=self._device)
                .repeat_interleave(2)
                .expand(self._num_envs // 4, self._max_thrusters)
                / 2
                * math.pi
                * 2
            )
            thrust_offset2 = (
                torch.arange(5, device=self._device)
                .repeat_interleave(2)
                .expand(self._num_envs // 4, self._max_thrusters)
                / 3
                * math.pi
                * 2
            )
            thrust_offset3 = (
                torch.arange(5, device=self._device)
                .repeat_interleave(2)
                .expand(self._num_envs // 4, self._max_thrusters)
                / 4
                * math.pi
                * 2
            )
            thrust_offset4 = (
                torch.arange(5, device=self._device)
                .repeat_interleave(2)
                .expand(self._num_envs // 4, self._max_thrusters)
                / 5
                * math.pi
                * 2
            )
            thrust_offset = torch.cat(
                [thrust_offset1, thrust_offset2, thrust_offset3, thrust_offset4], 0
            )
            # Generates a mask indicating if the thrusters are usable or not. Used by the transformer to mask the sequence.
            mask1 = torch.ones(
                (self._num_envs // 4, self._max_thrusters), device=self._device
            )
            mask2 = torch.ones(
                (self._num_envs // 4, self._max_thrusters), device=self._device
            )
            mask3 = torch.ones(
                (self._num_envs // 4, self._max_thrusters), device=self._device
            )
            mask4 = torch.ones(
                (self._num_envs // 4, self._max_thrusters), device=self._device
            )
            # TODO document this
            mask1[:, 4:] = 0
            mask2[:, 6:] = 0
            mask3[:, 8:] = 0
            mask = torch.cat([mask1, mask2, mask3, mask4], 0)
        else:
            # Generates N, two by two thruster
            thrust_offset = (
                torch.arange(self.thruster_cfg.num_anchors, device=self._device)
                .repeat_interleave(2)
                .expand(self._num_envs, self._max_thrusters)
                / self.thruster_cfg.num_anchors
                * math.pi
                * 2
            )
            # Generates a mask indicating if the thrusters are usable or not. Used by the transformer to mask the sequence.
            mask = torch.ones(
                (self._num_envs, self._max_thrusters), device=self._device
            )

        # ====================
        # Random thruster killing
        # ====================

        # Kill thrusters:
        if self.rand_cfg.kill_thrusters:
            # Generates 0 and 1 to decide how many thrusters will be killed
            weights = torch.ones((self._num_envs, 2), device=self._device)
            kills = torch.multinomial(
                weights, num_samples=self.rand_cfg.max_thruster_kill, replacement=True
            )
            # Selects L indices to set to N+1
            weights = torch.ones(self._max_thrusters, device=self._device).expand(
                self._num_envs, -1
            )
            kill_ids = torch.multinomial(
                weights, num_samples=self.rand_cfg.max_thruster_kill, replacement=False
            )
            # Multiplies kill or not kill with the ids.
            # If no kill, then the value is set to max_thrusters + 1, such that it can be filtered out later
            final_kill_ids = kills * kill_ids + (1 - kills) * self._max_thrusters
            # Creates a mask from the kills:
            kill_mask = torch.sum(
                torch.nn.functional.one_hot(final_kill_ids, self._max_thrusters + 1),
                dim=1,
            )
            # Removes the duplicates
            kill_mask = 1 - kill_mask[:, : self._max_thrusters]

            if self.thruster_cfg.use_four_configurations:
                mask[self._num_envs // 4 :] = (
                    mask[self._num_envs // 4 :] * kill_mask[self._num_envs // 4 :]
                )
            else:
                mask = mask * kill_mask

        # Generates the transforms and masks
        transforms2D = torch.zeros_like(self.transforms2D)  # Used to project the forces
        action_masks = torch.zeros_like(self.action_masks)  # Used to mask actions
        current_transforms = torch.zeros_like(
            self.current_transforms
        )  # Used to feed to the transformer

        # ====================
        # Randomizes the thruster poses and characteristics.
        # ====================

        # Randomizes the thrust force:
        if self.rand_cfg.randomize_thrust_force:
            thrust_force = (
                torch.rand((self._num_envs, self._max_thrusters), device=self._device)
                * (self.rand_cfg.max_thrust_force - self.rand_cfg.min_thrust_force)
                + self.rand_cfg.min_thrust_force
            )
        else:
            thrust_force = torch.ones(
                (self._num_envs, self._max_thrusters), device=self._device
            )

        # Thruster angular position with regards to the center of mass.
        theta2 = random_offset + thrust_offset
        # Randomizes thruster poses if requested:
        if self.rand_cfg.randomize_thruster_position:
            radius = self.core_cfg.radius * (
                1
                + torch.rand((self._num_envs, self._max_thrusters), device=self._device)
                * (self.rand_cfg.max_random_radius + self.rand_cfg.min_random_radius)
                - self.rand_cfg.min_random_radius
            )
            theta2 += (
                torch.rand((self._num_envs, self._max_thrusters), device=self._device)
                * (self.rand_cfg.random_theta * 2)
                - self.rand_cfg.random_theta
            )
        else:
            radius = self.core_cfg.radius
        # Thruster angle:
        theta = theta2 + thrust_90

        # ====================
        # Computes the 2D transforms of the thruster locations.
        # ====================

        # 2D transforms defining the thruster locations.
        transforms2D[:, :, 0, 0] = torch.cos(theta) * mask
        transforms2D[:, :, 0, 1] = torch.sin(-theta) * mask
        transforms2D[:, :, 1, 0] = torch.sin(theta) * mask
        transforms2D[:, :, 1, 1] = torch.cos(theta) * mask
        transforms2D[:, :, 2, 0] = torch.cos(theta2) * radius * mask
        transforms2D[:, :, 2, 1] = torch.sin(theta2) * radius * mask
        transforms2D[:, :, 2, 2] = 1 * mask
        # Actions masks to define which thrusters can be used.
        action_masks[:, :] = 1 - mask.long()
        # Transforms to feed to the transformer.
        current_transforms[:, :, 0] = torch.cos(theta) * mask
        current_transforms[:, :, 1] = torch.sin(-theta) * mask
        current_transforms[:, :, 2] = torch.cos(theta2) * radius * mask
        current_transforms[:, :, 3] = torch.sin(theta2) * radius * mask
        current_transforms[:, :, 4] = thrust_force * mask

        # Applies random permutations to the thrusters while keeping the non-used thrusters at the end of the sequence.
        if self.rand_cfg.random_permutation:
            weights = torch.ones(self._max_thrusters, device=self._device).expand(
                self._num_envs, -1
            )
            selected_thrusters = torch.multinomial(
                weights, num_samples=self._max_thrusters, replacement=False
            )
            mask = torch.gather(1 - mask, 1, selected_thrusters)
            _, sorted_idx = mask.sort(1)
            selected_thrusters = torch.gather(selected_thrusters, 1, sorted_idx)

            transforms2D = torch.gather(
                transforms2D,
                1,
                selected_thrusters.view(
                    self._num_envs, self._max_thrusters, 1, 1
                ).expand(self._num_envs, self._max_thrusters, 3, 3),
            )
            current_transforms = torch.gather(
                current_transforms,
                1,
                selected_thrusters.view(self._num_envs, self._max_thrusters, 1).expand(
                    self._num_envs, self._max_thrusters, 5
                ),
            )
            action_masks = torch.gather(action_masks, 1, selected_thrusters)
            thrust_force = torch.gather(thrust_force, 1, selected_thrusters)

        # Updates the proper indices
        self.thrust_force[env_ids] = thrust_force[env_ids]
        self.action_masks[env_ids] = action_masks[env_ids]
        self.current_transforms[env_ids] = current_transforms[env_ids]
        self.transforms2D[env_ids] = transforms2D[env_ids]

    def visualize(self, save_path: str = None):
        """
        Visualizes the thruster configuration."""

        from matplotlib import pyplot as plt
        from matplotlib import cm
        import numpy as np

        # Creates a list of color
        cmap = cm.get_cmap("hsv")
        colors = []
        for i in range(self._max_thrusters):
            colors.append(cmap(i / self._max_thrusters))

        # Split into 1/4th of the envs, so that we can visualize all the configs in use_four_configuration mode.
        env_ids = [
            0,
            1,
            2,
            3,
            self._num_envs // 4,
            self._num_envs // 4 + 1,
            self._num_envs // 4 + 2,
            self._num_envs // 4 + 3,
            2 * self._num_envs // 4,
            2 * self._num_envs // 4 + 1,
            2 * self._num_envs // 4 + 2,
            2 * self._num_envs // 4 + 3,
            3 * self._num_envs // 4,
            3 * self._num_envs // 4 + 1,
            3 * self._num_envs // 4 + 2,
            3 * self._num_envs // 4 + 3,
        ]

        # Generates a thrust on all the thrusters
        forces = torch.ones(
            (self._num_envs, self._max_thrusters),
            device=self._device,
            dtype=torch.float32,
        )
        # Project
        p, f = self.project_forces(forces)
        # Reshape and get only the 2D values for plot.
        p = p.reshape(self._num_envs, self._max_thrusters, 3)[:, :, :2]
        f = f.reshape(self._num_envs, self._max_thrusters, 3)[:, :, :2]
        p = np.array(p.cpu())
        f = np.array(f.cpu())

        # Plot
        fig, axs = plt.subplots(4, 4)
        fig.set_size_inches(20, 20)
        for i in range(4):
            for j in range(4):
                idx = env_ids[i * 4 + j]
                axs[i, j].quiver(
                    p[idx, :, 0],
                    p[idx, :, 1],
                    f[idx, :, 0],
                    f[idx, :, 1],
                    color=colors,
                    scale=4,
                    scale_units="xy",
                    angles="xy",
                )
                axs[i, j].set_xlim([-0.75, 0.75])
                axs[i, j].set_ylim([-0.75, 0.75])
        fig.tight_layout()
        fig.savefig(save_path, dpi=300)
        plt.close()
