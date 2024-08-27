from omni.isaac.kit import SimulationApp
import omni

from omniisaacgymenvs.envs.Physics.Hydrodynamics import Hydrodynamics
from omniisaacgymenvs.envs.Physics.Hydrostatics import Hydrostatics, get_euler_angles
from omniisaacgymenvs.envs.Physics.ThrusterDynamics import DynamicsFirstOrder

import pkgutil
import os
oige_path = os.path.dirname(pkgutil.get_loader("omniisaacgymenvs").get_filename())

import torch
import yaml
import numpy as np
import csv

config_path = os.path.join(oige_path, "cfg/config.yaml")


cfg = {
        "headless": False,
    }
simulation_app = SimulationApp(cfg)

# with open(config_path, "r") as file:
#     config = yaml.safe_load(file)
# from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig
# sim_config = SimConfig(config) # Loading this configuration, the robot does not move.

from omni.isaac.core import World

from heron import Heron
from heron_view import HeronView

task_cfg_path = os.path.join(oige_path, "cfg/task/ASV/GoThroughPosition.yaml")
with open(task_cfg_path, "r") as f:
    task_cfg = yaml.safe_load(f)
dt = task_cfg["sim"]["dt"]
controlFrequencyInv = task_cfg["env"]["controlFrequencyInv"]


timeline = omni.timeline.get_timeline_interface()
world = World(stage_units_in_meters=1.0, physics_dt=dt)
# https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.core/docs/index.html#omni.isaac.core.world.World.set_simulation_dt
world.scene.add_default_ground_plane()

heron = Heron(prim_path="/World/envs/heron", name="Kiki", translation=[-0.02942, 0, 0.18])
heron_view = HeronView(prim_paths_expr="/World/envs/heron")

cfg_path = os.path.join(oige_path, "cfg/task/robot/ASV_kingfisher.yaml")


with open(cfg_path, "r") as f:
    kingfisher_cfg = yaml.safe_load(f)


num_envs = 1
device = "cpu"
gravity = task_cfg["sim"]["gravity"][2]

hydrostatics_cfg = kingfisher_cfg["dynamics"]["hydrostatics"]
hydrostatics = Hydrostatics(num_envs=num_envs,device=device,gravity=gravity,params=hydrostatics_cfg)

hydrodynamics_cfg = kingfisher_cfg["dynamics"]["hydrodynamics"]
hydrodynamics_dr_cfg = kingfisher_cfg["asv_domain_randomization"]["drag"]
hydrodynamics = Hydrodynamics(dr_params=hydrodynamics_dr_cfg, num_envs=num_envs, device=device, params=hydrodynamics_cfg)

dr_thruster_cfg = kingfisher_cfg["asv_domain_randomization"]["thruster"]
dr_thruster_cfg["use_thruster_randomization"] = False
thruster_cfg = kingfisher_cfg["dynamics"]["thrusters"]
thurster_dynamics = DynamicsFirstOrder(
    dr_params=dr_thruster_cfg,
    num_envs=num_envs,
    device=device,
    timeConstant=thruster_cfg["timeConstant"],
    dt=dt,
    params=thruster_cfg
)

# save configuration used to a file
# with open("output/cfg.yaml", "w") as f:
#     yaml.dump(kingfisher_cfg, f)
#     yaml.dump(task_cfg, f)


thrust_cmds = torch.tensor([[0, 2]], dtype=torch.float32, device=device)
# thrust_cmds = torch.tensor([0.0, 0.0], dtype=torch.float32, device=device)
thrusters = torch.zeros((num_envs, 6), device=device, dtype=torch.float32)

world.scene.add(heron_view.base)
world.scene.add(heron_view.thruster_left)
world.scene.add(heron_view.thruster_right)
heron_view.base.set_masses([hydrostatics_cfg["mass"]])


timeline.play()
np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)


# Iterate over the CSV files in the input folder
input_folder = os.path.join(oige_path, "tests/kingfisher/input")
for file in os.listdir(input_folder):
    # Skip non-csv files
    if not file.endswith(".csv"):
        continue
    input_path = os.path.join(input_folder, file)

    input_file = open(input_path, "r")
    csv_input = csv.DictReader(input_file)

    output_path = input_path.replace("input", "output")
    output_file = open(output_path, "w")
    csv_output = csv.writer(output_file)
    csv_output.writerow(["time","thr_r","thr_l","pos_x","pos_y","lin_x","lin_y","ang_z","roll","pitch","yaw"])


    print(f"Processing file: {file}")
    world.reset()
    iter = 0

    # Warmup iterations
    for i in range(100):
        # GET POSITIONS and VELOCITIES
        root_pos, root_quats = heron_view.base.get_world_poses(clone=True)
        root_pos = torch.tensor(root_pos, dtype=torch.float32, device=device)
        root_quats = torch.tensor(root_quats, dtype=torch.float32, device=device)

        root_vels = heron_view.base.get_velocities(clone=True)
        root_vels = torch.tensor(root_vels, dtype=torch.float32, device=device)


        # COMPUTE FORCES
        hydrostatic_force = hydrostatics.compute_archimedes_metacentric_local(root_pos, root_quats)
        hydrodynamic_force = hydrodynamics.ComputeHydrodynamicsEffects(root_quats, root_vels)

        thrust_cmds[:, 0] = 0
        thrust_cmds[:, 1] = 0
        thrust_cmds[:, :] = torch.clamp(thrust_cmds, -0.99, 1.0)
        thurster_dynamics.set_target_force(thrust_cmds)
        thrusters[:, :] = thurster_dynamics.update_forces()


        # APPLY FORCES
        forces = hydrostatic_force[:, :3] + hydrodynamic_force[:, :3]
        torques = hydrostatic_force[:, 3:] + hydrodynamic_force[:, 3:]
        heron_view.base.apply_forces_and_torques_at_pos(forces=forces, torques=torques, is_global=False)
        heron_view.thruster_left.apply_forces_and_torques_at_pos(forces=thrusters[:, :3], is_global=False)
        heron_view.thruster_right.apply_forces_and_torques_at_pos(forces=thrusters[:, 3:], is_global=False)
        if i % controlFrequencyInv == 0:
            world.step(render=True)
        else:
            world.step(render=False)

    # Main loop
    start_time = world.current_time
    for row in csv_input:

        # GET POSITIONS and VELOCITIES
        root_pos, root_quats = heron_view.base.get_world_poses(clone=True)
        root_pos = torch.tensor(root_pos, dtype=torch.float32, device=device)
        root_quats = torch.tensor(root_quats, dtype=torch.float32, device=device)
        root_euler = get_euler_angles(root_quats)

        root_vels = heron_view.base.get_velocities(clone=True)
        root_vels = torch.tensor(root_vels, dtype=torch.float32, device=device)

        # TRANSFORM VELOCITIES TO LOCAL FRAME
        cos_theta = torch.cos(root_euler[:, 2])
        sin_theta = torch.sin(root_euler[:, 2])
        lin_vel_global =  root_vels[:, :2]
        lin_vel_local = torch.zeros_like(lin_vel_global)
        lin_vel_local[:, 0] = cos_theta * lin_vel_global[:, 0] + sin_theta * lin_vel_global[:, 1]
        lin_vel_local[:, 1] = -sin_theta * lin_vel_global[:, 0] + cos_theta * lin_vel_global[:, 1]

        # COMPUTE FORCES
        hydrostatic_force = hydrostatics.compute_archimedes_metacentric_local(root_pos, root_quats)
        hydrodynamic_force = hydrodynamics.ComputeHydrodynamicsEffects(root_quats, root_vels)
        forces = hydrostatic_force[:, :3] + hydrodynamic_force[:, :3]
        torques = hydrostatic_force[:, 3:] + hydrodynamic_force[:, 3:]

        thrust_cmds[:, 0] = float(row["thr_l"])
        thrust_cmds[:, 1] = float(row["thr_r"])
        thrust_cmds[:, :] = torch.clamp(thrust_cmds, -0.99, 1.0)
        thurster_dynamics.set_target_force(thrust_cmds)

        # UPDATE CSV OUTPUT VALUES
        time = world.current_time - start_time
        thr_r = row["thr_r"]
        thr_l = row["thr_l"]
        pos_x = root_pos[0, 0].item()
        pos_y = root_pos[0, 1].item()
        lin_x = lin_vel_local[0, 0].item()
        lin_y = lin_vel_local[0, 1].item()
        ang_z = root_vels[0, 5].item()
        roll = root_euler[0, -0].item()
        pitch = root_euler[0, 1].item()
        yaw = root_euler[0, 2].item()
        csv_output.writerow([time, thr_r, thr_l, pos_x, pos_y, lin_x, lin_y, ang_z, roll, pitch, yaw])

        # APPLY FORCES
        for i in range(controlFrequencyInv-1):
            thrusters[:, :] = thurster_dynamics.update_forces()
            heron_view.base.apply_forces_and_torques_at_pos(forces=forces, torques=torques, is_global=False)
            heron_view.thruster_left.apply_forces_and_torques_at_pos(forces=thrusters[:, :3], is_global=False)
            heron_view.thruster_right.apply_forces_and_torques_at_pos(forces=thrusters[:, 3:], is_global=False)
            world.step(render=False)
        thrusters[:, :] = thurster_dynamics.update_forces()
        heron_view.base.apply_forces_and_torques_at_pos(forces=forces, torques=torques, is_global=False)
        heron_view.thruster_left.apply_forces_and_torques_at_pos(forces=thrusters[:, :3], is_global=False)
        heron_view.thruster_right.apply_forces_and_torques_at_pos(forces=thrusters[:, 3:], is_global=False)
        world.step(render=True)


input_file.close()
output_file.close()

timeline.stop()
simulation_app.close()
