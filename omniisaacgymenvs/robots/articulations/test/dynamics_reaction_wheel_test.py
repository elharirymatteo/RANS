if __name__ == "__main__":
    from omni.isaac.kit import SimulationApp

    cfg = {
        "headless": False,
    }

    simulation_app = SimulationApp(cfg)
    from omni.isaac.core import World
    from pxr import UsdLux
    import omni
    import yaml
    import math
    from matplotlib import pyplot as plt
    import numpy as np


    import warp as wp



    from robots.actuators.dynamics import (
        Actuator
    )
    from omniisaacgymenvs.robots.articulations.utils.Types import DynamicsCfg, ZeroOrderDynamicsCfg, FirstOrderDynamicsCfg, SecondOrderDynamicsCfg, ActuatorCfg

    # Set the simulation parameters
    dt = 0.02  # Time step
    num_envs = 4  # Number of environments
    device = "cuda"  # Device for Warp computation
    wp.init()

    cfg = ActuatorCfg(dynamics={"name":"second_order", "natural_frequency":20.0, "damping_ratio":0.999}, limits={"limits":(-20, 20)}, scale_actions = False)
    actuator = Actuator(dt, num_envs, device, cfg)


    # Set initial actions for the reaction wheel
    action = wp.zeros((num_envs), device=device, dtype=wp.float32)
    mass = 0.25
    radius = 0.2
    moment_of_innertia = 0.5 * mass * radius * radius #0.5 * mass * radius^2

    # Arrays to store the actions and the resulting torque values
    actions = []
    ang_vels = []
    ang_acc = []
    torques = []

    # Run the simulation and record values over time
    for i in range(1000):
        if i < 333:
            action.fill_(1)  # Apply constant positive torque for the first 500 steps
            dynamics_dic = actuator.apply_dynamics(action)
            actions.append(action.to('cpu').numpy()[0])
            ang_vels.append(dynamics_dic["x"].to('cpu').numpy()[0])
            ang_acc.append(dynamics_dic["x_dot"].to('cpu').numpy()[0])
            torque = dynamics_dic["x_dot"].to('cpu').numpy()[0] * moment_of_innertia
            torques.append(torque)
        elif i < 666:
            action.fill_(-1)  # Apply constant negative torque for the next 500 steps
            dynamics_dic = actuator.apply_dynamics(action)
            actions.append(action.to('cpu').numpy()[0])
            ang_vels.append(dynamics_dic["x"].to('cpu').numpy()[0])
            ang_acc.append(dynamics_dic["x_dot"].to('cpu').numpy()[0])
            torque = dynamics_dic["x_dot"].to('cpu').numpy()[0] * moment_of_innertia
            torques.append(torque)
        else:
            action.fill_(0)  # Apply constant negative torque for the next 500 steps
            dynamics_dic = actuator.apply_dynamics(action)
            actions.append(action.to('cpu').numpy()[0])
            ang_vels.append(dynamics_dic["x"].to('cpu').numpy()[0])
            ang_acc.append(dynamics_dic["x_dot"].to('cpu').numpy()[0])
            torque = dynamics_dic["x_dot"].to('cpu').numpy()[0] * moment_of_innertia
            torques.append(torque)

    # Time array for plotting
    t = np.linspace(0, 1000*dt, 1000)

    # Plot the commanded actions and the resulting torque
    plt.figure()
    plt.title("Reaction Wheel Dynamics")
    plt.plot(t, actions, label="Commanded actions")
    # plt.plot(t, ang_acc, label="Angular Acceleration")
    plt.plot(t, ang_vels, label="Angular Velocities")
    plt.plot(t, torques, label="Torque applied")
    plt.xlabel("Time (s)")
    # plt.ylabel("Torque (Nm)")
    plt.legend()
    plt.savefig("/workspace/RANS/omniisaacgymenvs/robots/articulations/test/img_torque_dinam.png")



    #Second orderd
    dt = 0.0005
    num_envs = 100
    device = "cuda"
    wp.init()
    cfg = ActuatorCfg(dynamics={"name":"second_order", "natural_frequency":100.0, "damping_ratio":0.707}, limits={"limits":(-20, 20)}, scale_actions = False)
    actuator = Actuator(dt, num_envs, device, cfg)

    action = wp.zeros((num_envs), device=device, dtype=wp.float32)

    actions = []
    values = []

    for i in range(1000):
        if i < 500:
            action.fill_(0.5)
            output = actuator.apply_dynamics(action)
            values.append(output["x"].to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])
        else:
            action.fill_(-0.5)
            output = actuator.apply_dynamics(action)
            values.append(output["x"].to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])

    t = np.linspace(0, 1000*dt, 1000)
    plt.figure()
    plt.title("Second order dynamics")
    plt.plot(t, actions, label="cmd")
    plt.plot(t, values, label="actuator")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees ($^\circ$)")
    plt.savefig("/workspace/RANS/omniisaacgymenvs/robots/articulations/test/img_second_order.png")


    #First orderd
    dt = 0.005
    cfg = ActuatorCfg(dynamics={"name":"first_order", "time_constant":0.2}, limits={"limits":(0, 100)}, scale_actions = False)
    actuator = Actuator(dt, num_envs, device, cfg)

    action = wp.zeros((num_envs), device=device, dtype=wp.float32)

    actions = []
    values = []
    for i in range(1000):
        if i < 500:
            action.fill_(0.5)
            output = actuator.apply_dynamics(action)
            values.append(output.to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])
        else:
            action.fill_(-0.5)
            output = actuator.apply_dynamics(action)
            values.append(output.to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])

    t = np.linspace(0, 1000*dt, 1000)
    plt.figure()
    plt.title("First order dynamics")
    plt.plot(t, actions, label="cmd")
    plt.plot(t, values, label="actuator")
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust percentage (%)")
    plt.legend()
    plt.savefig("/workspace/RANS/omniisaacgymenvs/robots/articulations/test/img_first_order.png")
