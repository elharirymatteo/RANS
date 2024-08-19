from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

from typing import Optional


class HeronView(ArticulationView):
        def __init__(
            self, prim_paths_expr: str, name: Optional[str] = "HeronPlatformView"
        ) -> None:
            """[summary]"""

            super().__init__(
                prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False
            )
            self.base = RigidPrimView(
                prim_paths_expr=f"/World/envs/heron/base_link",
                name="base_view",
                reset_xform_properties=False,
            )

            self.thruster_left = RigidPrimView(
                prim_paths_expr=f"/World/envs/heron/thruster_left",
                name="thruster_left",
                reset_xform_properties=False,
            )
            self.thruster_right = RigidPrimView(
                prim_paths_expr=f"/World/envs/heron/thruster_right",
                name="thruster_right",
                reset_xform_properties=False,
            )
