__author__ = "Matteo El Hariry, Antoine Richard"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class DualArmRobotView(ArticulationView):
    def __init__(
        self, 
        prim_paths_expr: str = "/World/envs/.*/DualArmRobot", 
        name: Optional[str] = "DualArmRobotView", 
        track_contact_force: bool = False,
    ) -> None:
        """Initialize the DualArmRobot view."""

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )
        self.base = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/DualArmRobot/base",
            name="base_view",
            track_contact_forces=track_contact_force,
        )
        self.links = [
            RigidPrimView(
                prim_paths_expr=f"/World/envs/.*/DualArmRobot/link_{i+1}",
                name=f"link_{i+1}_view",
                track_contact_forces=track_contact_force,
            )
            for i in range(3)
        ]
        self.end_effectors = [
            RigidPrimView(
                prim_paths_expr=f"/World/envs/.*/DualArmRobot/end_effector_{i+1}",
                name=f"end_effector_{i+1}_view",
                track_contact_forces=track_contact_force,
            )
            for i in range(2)
        ]
        self.CoM = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/DualArmRobot/movable_CoM/CoM",
            name="CoM_view",
        )

    def get_link_indices(self):
        """Retrieve indices of links for control or observation."""
        self.link_indices = [
            self.get_dof_index(f"link_{i+1}_joint")
            for i in range(3)
        ]

    def get_end_effector_indices(self):
        """Retrieve indices of end effectors for control or observation."""
        self.end_effector_indices = [
            self.get_dof_index(f"end_effector_{i+1}_joint")
            for i in range(2)
        ]


    def get_CoM_indices(self):
        self.CoM_shifter_indices = [
            self.get_dof_index("com_x_axis_joint"),
            self.get_dof_index("com_y_axis_joint"),
        ]

    def get_plane_lock_indices(self):
        self.lock_indices = [
            self.get_dof_index("dar_world_joint_x"),
            self.get_dof_index("dar_world_joint_y"),
            self.get_dof_index("dar_world_joint_z"),
        ]
