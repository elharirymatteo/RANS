import matplotlib.pyplot as plt
import numpy as np
import os 
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from pathlib import Path


def plot_one_episode(ep_data, actions, save_dir):
    """
    Plot episode data for a single agent
    :param ep_data: dictionary containing episode data
    :param save_dir: directory where to save the plots
    :param all_agents: if True, plot average results over all agents, if False only the first agent is plotted
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)

    control_history = actions

    state_history = np.array([step["state"].cpu().numpy().flatten() for step in ep_data])


    # save data to csv file
    pd.DataFrame.to_csv(pd.DataFrame(control_history), save_dir + 'actions.csv')
    # setting the right task_data lavels based on the task flag.
    task_flag = state_history[0, 5].astype(int)
    if task_flag == 0:
        task_data_label = ['error_x', 'error_y']
    elif task_flag == 1:
        task_data_label = ['error_x', 'error_y', 'cos_error_heading', 'sin_error_heading']
    elif task_flag == 2:
        task_data_label = ['error_vx', 'error_vy']
    elif task_flag == 3:
        task_data_label = ['error_vx', 'error_vy', 'error_omega']
    else:
        task_data_label = []
    df_cols = ['cos_theta','sin_theta', 'lin_vel_x', 'lin_vel_y', 'ang_vel_z', 'task_flag'] + task_data_label
    pd.DataFrame.to_csv(pd.DataFrame(state_history[:, :8], columns=df_cols), save_dir + 'states_episode.csv')

    tgrid = np.linspace(0, len(control_history), len(control_history))
    fig_count = 0

    # °°°°°°°°°°°°°°°°°°°°°°°° plot linear speeds in time °°°°°°°°°°°°°°°°°°°°°°°°°
    lin_vels = state_history[:, 2:4]
    # plot linear velocity
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, lin_vels[:, 0], 'r-')
    plt.plot(tgrid, lin_vels[:, 1], 'g-')
    plt.xlabel('Time steps')
    plt.ylabel('Velocity [m/s]')
    plt.legend(['x', 'y'], loc='best')
    plt.title('Velocity state history')
    plt.grid()
    plt.savefig(save_dir + '_lin_vel')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot angular speeds in time °°°°°°°°°°°°°°°°°°°°°°°°°
    ang_vel_z = state_history[:, 4:5]
    # plot angular speed (z coordinate)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, ang_vel_z, 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Angular speed [rad/s]')
    plt.legend(['z'], loc='best')
    plt.title('Angular speed state history')
    plt.grid()
    plt.savefig(save_dir + '_ang_vel')
    #plt.show()
    # °°°°°°°°°°°°°°°°°°°°°°°° plot heading cos, sin °°°°°°°°°°°°°°°°°°°°°°°°°
    headings = state_history[:, :2]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, headings[:, 0], 'r-') # cos
    plt.plot(tgrid, headings[:, 1], 'g-') # sin
    plt.xlabel('Time steps')
    plt.ylabel('Heading')
    plt.legend(['cos(${\\theta}$)', 'sin(${\\theta}$)'], loc='best')
    plt.title('Heading state history')
    plt.grid()
    plt.savefig(save_dir + '_heading')


    # °°°°°°°°°°°°°°°°°°°°°°°° plot absolute heading angle °°°°°°°°°°°°°°°°°°°°°°°°°
    headings = state_history[:, :2]
    angles = np.arctan2(headings[:, 0], headings[:, 1])
    #angles = np.where(angles < 0, angles + np.pi, angles)
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, angles, 'c-')
    plt.xlabel('Time steps')
    plt.ylabel('Angle [rad]')
    plt.legend(['${\\theta}$'], loc='best')
    plt.title('Angle state history')
    plt.grid()
    plt.savefig(save_dir + '_angle')
# Compute the angle using numpy.arctan2

    # °°°°°°°°°°°°°°°°°°°°°°°° plot actions in time °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = np.array(control_history)
    colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b-', 'g-', 'r-', 'c-']
    for k in range(control_history.shape[1]):
        col = colours[k % control_history.shape[0]]
        plt.step(tgrid, control_history[:, k], col)
    plt.xlabel('Time steps')
    plt.ylabel('Control [N]')
    plt.legend([f'u{k}' for k in range(control_history.shape[1])], loc='best')
    plt.title('Thrust control')
    plt.grid()
    plt.savefig(save_dir + '_actions')
    
        # °°°°°°°°°°°°°°°°°°°°°°°° plot actions histogram °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = np.array(control_history)
    n_bins = len(control_history[0])  
    minor_locator = AutoMinorLocator(1)
    plt.gca().xaxis.set_minor_locator(minor_locator) 
    n, bins, patches = plt.hist(np.sum(control_history, axis=1), bins=len(control_history[0]), edgecolor='white')

    xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
    ticklabels = [f'T{i+1}' for i in range(n_bins)]
    plt.xticks(xticks, ticklabels)
    plt.yticks([])
    for idx, value in enumerate(n):
        if value > 0:
            plt.text(xticks[idx], value, int(value), ha='center')
    plt.title('Number of thrusts in episode')
    
    plt.savefig(save_dir + '_actions_hist')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot rewards °°°°°°°°°°°°°°°°°°°°°°°°°
    # fig_count += 1
    # plt.figure(fig_count)
    # plt.clf()
    # plt.plot(tgrid, reward_history, 'b-')
    # plt.xlabel('Time steps')
    # plt.ylabel('Reward')
    # plt.legend(['reward'], loc='best')
    # plt.title('Reward history')
    # plt.grid()
    # # plt.show()
    # plt.savefig(save_dir + '_reward')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot x, y position error over time °°°°°°°°°°°°°°°°°°°°°°°°°
    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, pos_error[:, 0], 'r-')
    plt.plot(tgrid, pos_error[:, 1], 'g-')
    plt.xlabel('Time steps')
    plt.ylabel('Position [m]')
    plt.legend(['x position', 'y position'], loc='best')
    plt.title('Planar position')
    plt.grid()
    plt.savefig(save_dir + '_pos_error')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot x, y position in plane °°°°°°°°°°°°°°°°°°°°°°°°°
    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(pos_error[:, 0], pos_error[:, 1])
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Planar position')
    plt.grid()
    plt.savefig(save_dir + '_pos_xy_plane')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot distance to target over time °°°°°°°°°°°°°°°°°°°°°°°°°
    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0), 'c')
    plt.xlabel('Time steps')
    plt.ylabel('Distance [m]')
    plt.legend(['abs dist'], loc='best')
    plt.title('Distance to target')
    plt.grid()
    plt.savefig(save_dir + '_dist_to_target')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot log-distance to target over time °°°°°°°°°°°°°°°°°°°°°°°°°
    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale('log')
    plt.plot(tgrid, np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0), 'm')
    plt.xlabel('Time steps')
    plt.ylabel('Log distance [m]')
    plt.legend(['x-y dist'], loc='best')
    plt.title('Log distance to target')
    plt.grid(True)
    plt.savefig(save_dir + '_log_dist_to_target')


if __name__ == "__main__":

    load_dir = Path("./lab_tests/new_mass/")
    sub_dirs = [d for d in load_dir.iterdir() if d.is_dir()]
    if sub_dirs:
        latest_exp = max(sub_dirs, key=os.path.getmtime)
        print("Plotting data for experiment:", latest_exp)
        n_episodes = 1
    else:
        print("No experiments found in", load_dir)
        exit()
    obs_path = os.path.join(latest_exp, "obs.npy")
    actions_path = os.path.join(latest_exp, "act.npy")
    
    if not os.path.exists(obs_path) or not os.path.exists(actions_path):
        print("Required files not found in", latest_exp)
        exit()

    obs = np.load(obs_path, allow_pickle=True)
    actions = np.load(actions_path)

    save_to = os.path.join(latest_exp, 'plots/')
    os.makedirs(save_to, exist_ok=True)

    plot_one_episode(obs, actions, save_to)

    print("Done!")



