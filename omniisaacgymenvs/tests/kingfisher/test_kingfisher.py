from omni.isaac.kit import SimulationApp
import omni

from omniisaacgymenvs.envs.Physics.Hydrodynamics import Hydrodynamics
from omniisaacgymenvs.envs.Physics.Hydrostatics import Hydrostatics
from omniisaacgymenvs.envs.Physics.ThrusterDynamics import DynamicsFirstOrder

import torch
import yaml
import numpy as np


cfg = {"headless": False}
simulation_app = SimulationApp(cfg)

from omni.isaac.core import World

from heron import Heron
from heron_view import HeronView


timeline = omni.timeline.get_timeline_interface()
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()


heron = Heron(prim_path="/World/envs/heron", name="Kiki", translation=[0, 0, 0.11])
heron_view = HeronView(prim_paths_expr="/World/envs/heron")

cfg_path = "/home/luis/workspaces/RANS/omniisaacgymenvs/cfg/task/robot/ASV_kingfisher.yaml"
with open(cfg_path, "r") as f:
    kingfisher_cfg = yaml.safe_load(f)

num_envs = 1
device = "cpu"
gravity = 9.81

hydrostatics_cfg = kingfisher_cfg["dynamics"]["hydrostatics"]
hydrostatics = Hydrostatics(num_envs=num_envs,device=device,gravity=gravity,params=hydrostatics_cfg)

hydrodynamics_cfg = kingfisher_cfg["dynamics"]["hydrodynamics"]
hydrodynamics_dr_cfg = kingfisher_cfg["asv_domain_randomization"]["drag"]
hydrodynamics = Hydrodynamics(dr_params=hydrodynamics_dr_cfg, num_envs=num_envs, device=device, params=hydrodynamics_cfg)

world.scene.add(heron_view.base)
world.scene.add(heron_view.thruster_left)
world.scene.add(heron_view.thruster_right)

# with open("cfg/config.yaml", "r") as file:
#     config = yaml.safe_load(file)
# from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
# sim_config = SimConfig(config) # Loading this configuration, the robot does not move.

world.reset()
timeline.play()
np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)


iter = 0

while simulation_app.is_running():
    iter += 1
    root_pos, root_quats = heron_view.base.get_world_poses(clone=True)
    print(f"Root position: {root_pos}")
    print(f"Root quaternions: {root_quats}")
    # Transform the position and rotation to torch tensors
    root_pos = torch.tensor(root_pos, dtype=torch.float32, device=device)
    root_quats = torch.tensor(root_quats, dtype=torch.float32, device=device)

    root_vels = heron_view.base.get_velocities(clone=True)
    root_vels = torch.tensor(root_vels, dtype=torch.float32, device=device)
    print(f"Root velocities: {root_vels}")

    hydrostatic_force = hydrostatics.compute_archimedes_metacentric_local(root_pos, root_quats)
    print(f"hydrostatic_force: {hydrostatic_force}")

    hydrodynamic_force = hydrodynamics.ComputeHydrodynamicsEffects(root_quats, root_vels)
    print(f"hydrodynamic_force: {hydrodynamic_force}")

    forces = -hydrostatic_force[:, :3] + hydrodynamic_force[:, :3]
    torques = hydrostatic_force[:, 3:] + hydrodynamic_force[:, 3:]
    print(f"total forces: {forces}")
    print(f"total torques: {torques}")


    # heron_view.base.apply_forces_and_torques_at_pos(force
    heron_view.base.apply_forces_and_torques_at_pos(forces=forces, torques=torques, is_global=False)
    if iter >= 500 and iter < 1500:
        thurster_force = torch.tensor([[20, 0, 0]], dtype=torch.float32, device=device)
        print(f"thurster_force: {thurster_force}")

        heron_view.thruster_right.apply_forces_and_torques_at_pos(
            forces=thurster_force, is_global=False
        )

    print()
    world.step(render=True)

timeline.stop()
simulation_app.close()
