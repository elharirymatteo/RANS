import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_episode_data_virtual(ep_data, save_dir, all_agents=False):
    """
    Plot episode data for a single agent
    :param ep_data: dictionary containing episode data
    :param save_dir: directory where to save the plots
    :param all_agents: if True, plot average results over all agents, if False only the first agent is plotted
    :return:
    """
    # TODO: place all the different plots in a separate function, to be called from here based on the episode (best, worst, first etc.)
    reward_history = ep_data['rews']
    print(f'reward_history.shape[1]: {reward_history.shape[1]}')
    all_agents = False if reward_history.shape[1] == 1 else all_agents
    # plot average results over all agents

    control_history = ep_data['act']
    reward_history = ep_data['rews']
    info_history = ep_data['info']
    state_history = ep_data['obs']

    if all_agents:
        best_agent = np.argmax(reward_history.sum(axis=0))
        worst_agent = np.argmin(reward_history.sum(axis=0))
        rand_agent = np.random.choice(list(set(range(0, reward_history.shape[1]))-set([best_agent, worst_agent])))
        print('Best agent: ', best_agent, '| Worst agent: ', worst_agent, '| Random Agent', rand_agent)
        # plot best and worst episodes data
        plot_one_episode({k:np.array([v[best_agent] for v in vals]) for k,vals in ep_data.items()}, save_dir+"best_ep/")
        plot_one_episode({k:np.array([v[worst_agent] for v in vals]) for k,vals in ep_data.items()}, save_dir+"worst_ep/")
        plot_one_episode({k:np.array([v[rand_agent] for v in vals]) for k,vals in ep_data.items()}, save_dir+f'rand_ep_{rand_agent}/')
        
        all_distances = ep_data['all_dist']

        # °°°°°°°°°°°°°°°°°°°°°°°° plot meand and std distance °°°°°°°°°°°°°°°°°°°°°°°°°
        tgrid = np.linspace(0, len(all_distances), len(control_history))
        fig_count = 0
        fig, ax = plt.subplots()
        ax.plot(tgrid, all_distances.mean(axis=1), alpha=0.5, color='blue', label='mean_dist', linewidth = 2.0)
        ax.fill_between(tgrid, all_distances.mean(axis=1) - all_distances.std(axis=1), all_distances.mean(axis=1) 
                        + all_distances.std(axis=1), color='blue', alpha=0.4)
        ax.fill_between(tgrid, all_distances.mean(axis=1) - 2*all_distances.std(axis=1), all_distances.mean(axis=1) 
                        + 2*all_distances.std(axis=1), color='blue', alpha=0.2)
        ax.plot(tgrid, all_distances[:, best_agent], alpha=0.5, color='green', label='best', linewidth = 2.0)
        ax.plot(tgrid, all_distances[:, worst_agent], alpha=0.5, color='red', label='worst', linewidth = 2.0)
        plt.xlabel('Time steps')
        plt.ylabel('Distance [m]')
        plt.legend(['mean',  'best', 'worst', '1-std', '2-std'], loc='best')
        plt.title(f'Mean, best and worst distances over {all_distances.shape[1]} episodes')
        plt.grid()
        plt.savefig(save_dir + '_mean_best_worst_dist')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot meand and std reward °°°°°°°°°°°°°°°°°°°°°°°°°
        fig_count += 1
        fig, ax = plt.subplots()

        ax.plot(tgrid, reward_history.mean(axis=1), alpha=0.5, color='blue', label='mean_dist', linewidth = 2.0)
        ax.fill_between(tgrid, reward_history.mean(axis=1) - reward_history.std(axis=1), reward_history.mean(axis=1) 
                        + reward_history.std(axis=1), color='blue', alpha=0.4)
        ax.fill_between(tgrid, reward_history.mean(axis=1) - 2*reward_history.std(axis=1), reward_history.mean(axis=1) 
                        + 2*reward_history.std(axis=1), color='blue', alpha=0.2)
        ax.plot(tgrid, reward_history[:, best_agent], alpha=0.5, color='green', label='best', linewidth = 2.0)
        ax.plot(tgrid, reward_history[:, worst_agent], alpha=0.5, color='red', label='worst', linewidth = 2.0)
        plt.xlabel('Time steps')
        plt.ylabel('Reward')
        plt.legend(['mean', 'best', 'worst', '1-std', '2-std'], loc='best')
        plt.title(f'Mean, best and worst reward over {all_distances.shape[1]} episodes')
        plt.grid()
        plt.savefig(save_dir + '_mean_best_worst_reward')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot meand and std angular speed °°°°°°°°°°°°°°°°°°°°°°°°°
        fig_count += 1
        fig, ax = plt.subplots()
        ang_vel_z = np.array(state_history[:, :, 4:5])[:,:,0]

        ax.plot(tgrid, ang_vel_z.mean(axis=1), alpha=0.5, color='blue', label='mean_dist', linewidth = 1.0)
        ax.fill_between(tgrid, ang_vel_z.mean(axis=1) - ang_vel_z.std(axis=1), ang_vel_z.mean(axis=1) 
                        + ang_vel_z.std(axis=1), color='blue', alpha=0.4)
        ax.fill_between(tgrid, ang_vel_z.mean(axis=1) - 2*ang_vel_z.std(axis=1), ang_vel_z.mean(axis=1) 
                        + 2*ang_vel_z.std(axis=1), color='blue', alpha=0.2)
        ax.plot(tgrid, ang_vel_z[:, best_agent], alpha=0.5, color='green', label='best', linewidth = 1.0)
        ax.plot(tgrid, ang_vel_z[:, worst_agent], alpha=0.5, color='red', label='worst', linewidth = 1.0)
        plt.xlabel('Time steps')
        plt.ylabel('Angular speed [rad/s]')
        plt.legend(['mean', 'best', 'worst', '1-std', '2-std'], loc='best')
        plt.title(f'Angular speed of mean, best and worst agents {ang_vel_z.shape[1]} episodes')
        plt.grid()
        plt.savefig(save_dir + '_mean_best_worst_ang_vel')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot mean actions histogram °°°°°°°°°°°°°°°°°°°°°°°°°
        fig_count += 1
        plt.figure(fig_count)
        plt.clf()
        control_history = control_history.reshape((control_history.shape[1], control_history.shape[0], control_history.shape[2]))
        control_history = np.array([c for c in control_history])

        freq = pd.DataFrame(data=np.array([control_history[i].sum(axis=0) for i in range(control_history.shape[0])]), 
                    columns=[f'T{i+1}' for i in range(control_history.shape[2])])
        mean_freq = freq.mean()
        plt.bar(mean_freq.index, mean_freq.values)
        plt.title(f'Mean number of thrusts in {control_history.shape[1]} episodes')
        plt.savefig(save_dir + '_mean_actions_hist')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot all the episodes trajectories in the 2d plane  °°°°°°°°°°°°°°°°°°°°°°°°°
      
        fig_count += 1
        plt.figure(fig_count)
        plt.clf()    
        colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        positions = state_history[:, :, 6:8]
        print(positions.shape)

        for j in range(positions.shape[1]):
            col = colours[j % len(colours)]
            plt.plot(positions[:, j, 0], positions[:, j, 1], color=col, alpha=0.5, linewidth=0.75)
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')

        plt.grid(alpha=0.3)
        plt.title(f'Trajectories in 2D space [{positions.shape[1]} episodes]')
        plt.gcf().dpi = 200
        plt.savefig(save_dir + '_multi_traj')
        # plt.show()

        return



def plot_one_episode(ep_data, save_dir):
    """
    Plot episode data for a single agent
    :param ep_data: dictionary containing episode data
    :param save_dir: directory where to save the plots
    :param all_agents: if True, plot average results over all agents, if False only the first agent is plotted
    :return:
    """
    os.makedirs(save_dir, exist_ok=True)

    control_history = ep_data['act']
    reward_history = ep_data['rews']
    info_history = ep_data['info']
    state_history = ep_data['obs']
    all_distances = ep_data['all_dist']

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
    control_history_df = pd.DataFrame(data=control_history)

    fig, axes = plt.subplots(len(control_history_df.columns), 1, sharex=True, figsize=(8, 6))
    # Select subset of colors from a colormap
    colormap = cm.get_cmap('tab20')
    num_colors = len(control_history_df.columns)
    colors = [colormap(i) for i in range(0, num_colors*2, 2)]
    for i, column in enumerate(control_history_df.columns):
        control_history_df[column].plot(ax=axes[i], color=colors[i])
        axes[i].set_ylabel(f'T{column}')
    fig.savefig(save_dir + '_actions')
    
        # °°°°°°°°°°°°°°°°°°°°°°°° plot actions histogram °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = np.array(control_history)

    actions_df = pd.DataFrame(control_history, columns=[f'T{i+1}' for i in range(control_history.shape[1])])
    freq = actions_df.sum()
    plt.bar(freq.index, freq.values)
    plt.title('Number of thrusts in episode')
    plt.tight_layout()
    plt.savefig(save_dir + '_actions_hist')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot rewards °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, reward_history, 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Reward')
    plt.legend(['reward'], loc='best')
    plt.title('Reward history')
    plt.grid()
    # plt.show()
    plt.savefig(save_dir + '_reward')

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
    x, y = pos_error[:, 0], pos_error[:, 1]
    fig,ax = plt.subplots(figsize=(6,6))
    
     #Setting the limit of x and y direction to define which portion to zoom
    x1, x2, y1, y2 = -.08, .08, -.1, .1
    axins = inset_axes(ax, width=2, height=1.5)
    ax.plot(x, y)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.plot(x, y)

    fig.savefig(save_dir + '_pos_xy_plane')
    
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

###################################################################################################################
###################################################################################################################

def plot_episode_data_old(ep_data, save_dir):

    control_history = ep_data['act']
    reward_history = ep_data['rews']
    info_history = ep_data['info']
    state_history = ep_data['obs']
    tgrid = np.linspace(0, len(control_history), len(control_history))
    fig_count = 0

    # °°°°°°°°°°°°°°°°°°°°°°°° plot linear speeds in time °°°°°°°°°°°°°°°°°°°°°°°°°
    lin_vels = state_history[:, 12:15]
    # plot linear velocity
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, lin_vels[:, 0], 'r-')
    plt.plot(tgrid, lin_vels[:, 1], 'g-')
    plt.plot(tgrid, lin_vels[:, 2], 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Velocity [m/s]')
    plt.legend(['x', 'y'], loc='best')
    plt.title('Velocity state history')
    plt.grid()
    plt.savefig(save_dir + '_lin_vel')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot angular speeds in time °°°°°°°°°°°°°°°°°°°°°°°°°
    ang_vels = state_history[:, 15:18]
    # plot angular speed (z coordinate)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    #plt.plot(tgrid, ang_vels[:, 0], 'r-')
    #plt.plot(tgrid, ang_vels[:, 1], 'g-')
    plt.plot(tgrid, ang_vels[:, 2], 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Angular speed [rad/s]')
    plt.legend(['z'], loc='best')
    plt.title('Angular speed state history')
    plt.grid()
    plt.savefig(save_dir + '_ang_vel')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot distance to target time °°°°°°°°°°°°°°°°°°°°°°°°°
    positions = state_history[:, :3]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, positions[:, 0], 'r-')
    plt.plot(tgrid, positions[:, 1], 'g-')
    plt.xlabel('Time steps')
    plt.ylabel('Position [m]')
    plt.legend(['x', 'y'], loc='best')
    plt.title('Position state history')
    plt.grid()
    plt.savefig(save_dir + '_pos')

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
    minor_locator = AutoMinorLocator(2)
    plt.gca().xaxis.set_minor_locator(minor_locator) 
    n, bins, patches = plt.hist(np.sum(control_history, axis=1), bins=len(control_history[0]), edgecolor='white')
    print(n, bins)
    xticks = [(bins[idx+1] + value)/2 for idx, value in enumerate(bins[:-1])]
    ticklabels = [f'T{i+1}' for i in range(n_bins)]
    plt.xticks(xticks, ticklabels)
    plt.yticks([])
    for idx, value in enumerate(n):
        if value > 0:
            plt.text(xticks[idx], value+5, int(value), ha='center')
    plt.title('Number of thrusts in episode')
    
    plt.savefig(save_dir + '_actions_hist')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot rewards °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, reward_history, 'b-')
    plt.xlabel('Time steps')
    plt.ylabel('Reward')
    plt.legend(['reward'], loc='best')
    plt.title('Reward history')
    plt.grid()
    # plt.show()
    plt.savefig(save_dir + '_reward')

    # °°°°°°°°°°°°°°°°°°°°°°°° plot dist to target °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    if 'episode' in info_history[0].keys() and ('position_error' and 'heading_error') in info_history[0]['episode'].keys():
        pos_error = np.array([info_history[j]['episode']['position_error'].cpu().numpy() for j in range(len(info_history))])
        head_error = np.array([info_history[j]['episode']['heading_error'].cpu().numpy() for j in range(len(info_history))])
        plt.figure(fig_count)
        plt.clf()

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Position error [m]', color=color)
        ax1.plot(tgrid, pos_error, color=color)
        ax1.grid()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Heading error [rad]', color=color)  # we already handled the x-label with ax1
        ax2.plot(tgrid, head_error, color=color)
        plt.title('Precision metrics')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #ax2.grid()
        plt.savefig(save_dir + '_distance_error')
    # °°°°°°°°°°°°°°°°°°°°°°°° plot dist to target rewards °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    if 'episode' in info_history[0].keys() and ('position_reward' and 'heading_reward') in info_history[0]['episode'].keys():
        pos_error = np.array([info_history[j]['episode']['position_reward'].cpu().numpy() for j in range(len(info_history))])
        head_error = np.array([info_history[j]['episode']['heading_reward'].cpu().numpy() for j in range(len(info_history))])
        plt.figure(fig_count)
        plt.clf()

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel('Position reward', color=color)
        ax1.plot(tgrid, pos_error, color=color)
        ax1.grid()
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        ax2.set_ylabel('Heading reward', color=color)  # we already handled the x-label with ax1
        ax2.plot(tgrid, head_error, color=color)
        plt.title('Rewards')
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #ax2.grid()
        plt.savefig(save_dir + '_rewards_infos')