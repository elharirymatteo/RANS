__author__ = "Antoine Richard, Matteo El Hariry"
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


class AGVSkidSteer2WView(ArticulationView):
    def __init__(
        self, prim_paths_expr: str, 
        name: Optional[str] = "AGV_SS_2W_View", 
        track_contact_force:bool = False,
    ) -> None:
        """[summary]"""

        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,
        )
        self.base = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/AGV_SS_2W/core",
            name="base_view",
            track_contact_forces=track_contact_force,
        )

    def get_wheel_indices(self):
        self.left_wheel_index =  self.get_dof_index("left_wheel")
        self.right_wheel_index = self.get_dof_index("right_wheel")