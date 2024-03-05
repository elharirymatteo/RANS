__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView


class ModularFloatingPlatformView(ArticulationView):
    def __init__(
        self, prim_paths_expr: str, name: Optional[str] = "ModularFloatingPlatformView"
    ) -> None:
        """[summary]"""

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )
        self.base = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/Modular_floating_platform/core/body",
            name="base_view",
        )
        self.CoM = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/Modular_floating_platform/movable_CoM/CoM",
            name="CoM_view",
        )
        self.thrusters = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/Modular_floating_platform/v_thruster_*",
            name="thrusters",
        )

    def get_CoM_indices(self):
        self.CoM_shifter_indices = [
            self.get_dof_index("x_axis_joint"),
            self.get_dof_index("y_axis_joint"),
            self.get_dof_index("z_axis_joint"),
        ]
