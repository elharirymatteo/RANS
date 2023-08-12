import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns


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
    # info_history = ep_data['info']
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
        ax.plot(tgrid, all_distances.mean(axis=1), alpha=0.5, color='blue', label='mean_dist', linewidth = 1.5)
        ax.fill_between(tgrid, all_distances.mean(axis=1) - all_distances.std(axis=1), all_distances.mean(axis=1) 
                        + all_distances.std(axis=1), color='blue', alpha=0.4)
        ax.fill_between(tgrid, all_distances.mean(axis=1) - 2*all_distances.std(axis=1), all_distances.mean(axis=1) 
                        + 2*all_distances.std(axis=1), color='blue', alpha=0.2)
        ax.plot(tgrid, all_distances[:, best_agent], alpha=0.5, color='green', label='best', linewidth = 1.5)
        ax.plot(tgrid, all_distances[:, worst_agent], alpha=0.5, color='red', label='worst', linewidth = 1.5)
        plt.xlabel('Time steps')
        plt.ylabel('Distance [m]')
        plt.legend(['mean',  'best', 'worst', '1-std', '2-std'], loc='best')
        plt.title(f'Mean, best and worst distances over {all_distances.shape[1]} episodes')
        plt.grid()
        plt.savefig(save_dir + '_mean_best_worst_dist')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot all distances °°°°°°°°°°°°°°°°°°°°°°°°°
        tgrid = np.linspace(0, len(all_distances), len(control_history))
        fig_count = 0
        fig, ax = plt.subplots()
        cmap = cm.get_cmap('tab20') 
        for j in range(all_distances.shape[1]):
            ax.plot(tgrid, all_distances[:, j], alpha=1., color=cmap(j % cmap.N), linewidth = 1.0)
        plt.xlabel('Time steps')
        plt.ylabel('Distance [m]')
        plt.title(f'All distances over {all_distances.shape[1]} episodes')
        plt.grid()
        plt.savefig(save_dir + '_all_dist')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot meand and std reward °°°°°°°°°°°°°°°°°°°°°°°°°
        fig_count += 1
        fig, ax = plt.subplots()

        ax.plot(tgrid, reward_history.mean(axis=1), alpha=0.5, color='blue', label='mean_dist', linewidth = 1.5)
        ax.fill_between(tgrid, reward_history.mean(axis=1) - reward_history.std(axis=1), reward_history.mean(axis=1) 
                        + reward_history.std(axis=1), color='blue', alpha=0.4)
        ax.fill_between(tgrid, reward_history.mean(axis=1) - 2*reward_history.std(axis=1), reward_history.mean(axis=1) 
                        + 2*reward_history.std(axis=1), color='blue', alpha=0.2)
        ax.plot(tgrid, reward_history[:, best_agent], alpha=0.5, color='green', label='best', linewidth = 1.5)
        ax.plot(tgrid, reward_history[:, worst_agent], alpha=0.5, color='red', label='worst', linewidth = 1.5)
        plt.xlabel('Time steps')
        plt.ylabel('Reward')
        plt.legend(['mean', 'best', 'worst', '1-std', '2-std'], loc='best')
        plt.title(f'Mean, best and worst reward over {all_distances.shape[1]} episodes')
        plt.grid()
        plt.savefig(save_dir + '_mean_best_worst_reward')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot meand and std angular speed °°°°°°°°°°°°°°°°°°°°°°°°°
        fig_count += 1
        fig, ax = plt.subplots()
        ang_vel_z = state_history[:, :, 4:5][:,:,0] # getting rid of the extra dimension

        ax.plot(tgrid, ang_vel_z.mean(axis=1), alpha=0.5, color='blue', label='mean_dist', linewidth = 1.5)
        ax.fill_between(tgrid, ang_vel_z.mean(axis=1) - ang_vel_z.std(axis=1), ang_vel_z.mean(axis=1) 
                        + ang_vel_z.std(axis=1), color='blue', alpha=0.4)
        ax.fill_between(tgrid, ang_vel_z.mean(axis=1) - 2*ang_vel_z.std(axis=1), ang_vel_z.mean(axis=1) 
                        + 2*ang_vel_z.std(axis=1), color='blue', alpha=0.2)
        ax.plot(tgrid, ang_vel_z[:, best_agent], alpha=0.5, color='green', label='best', linewidth = 1.5)
        ax.plot(tgrid, ang_vel_z[:, worst_agent], alpha=0.5, color='red', label='worst', linewidth = 1.5)
        plt.xlabel('Time steps')
        plt.ylabel('Angular speed [rad/s]')
        plt.legend(['mean', 'best', 'worst', '1-std', '2-std'], loc='best')
        plt.title(f'Angular speed of mean, best and worst agents {ang_vel_z.shape[1]} episodes')
        plt.grid()
        plt.savefig(save_dir + '_mean_best_worst_ang_vel')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot mean actions histogram °°°°°°°°°°°°°°°°°°°°°°°°°
        # fig_count += 1
        # plt.figure(fig_count)
        # plt.clf()
        # control_history = control_history.reshape((control_history.shape[1], control_history.shape[0], control_history.shape[2]))
        # control_history = np.array([c for c in control_history])

        # freq = pd.DataFrame(data=np.array([control_history[i].sum(axis=0) for i in range(control_history.shape[0])]), 
        #             columns=[f'T{i+1}' for i in range(control_history.shape[2])])
        # mean_freq = freq.mean()
        # plt.bar(mean_freq.index, mean_freq.values)
        # plt.title(f'Mean number of thrusts in {control_history.shape[0]} episodes')
        # plt.savefig(save_dir + '_mean_actions_hist')
        
        # °°°°°°°°°°°°°°°°°°°°°°°° plot mean actions boxplot °°°°°°°°°°°°°°°°°°°°°°°°°
        fig_count += 1
        plt.figure(fig_count)
        plt.clf()
        control_history = control_history.reshape((control_history.shape[1], control_history.shape[0], control_history.shape[2]))
        control_history = np.array([c for c in control_history])

        freq = pd.DataFrame(data=np.array([control_history[i].sum(axis=0) for i in range(control_history.shape[0])]), 
                    columns=[f'T{i+1}' for i in range(control_history.shape[2])])
        #mean_freq = freq.mean()
        sns.boxplot(data=freq, orient="h")
        plt.title(f'Mean number of thrusts in {control_history.shape[0]} episodes')
        plt.savefig(save_dir + '_actions_boxplot')

        # °°°°°°°°°°°°°°°°°°°°°°°° plot all the episodes trajectories in the 2d plane  °°°°°°°°°°°°°°°°°°°°°°°°°
      
        fig_count += 1
        plt.figure(fig_count)
        plt.clf()    
        positions = state_history[:, :, 6:8]
        print(positions.shape)

        cmap = cm.get_cmap('tab20') 
        for j in range(positions.shape[1]):
            col = cmap(j % cmap.N)  # Select a color from the colormap based on the current index
            plt.plot(positions[:, j, 0], positions[:, j, 1], color=col, alpha=1., linewidth=0.75)
            plt.xlabel('X [m]')
            plt.ylabel('Y [m]')

        plt.grid(alpha=0.3)
        plt.title(f'Trajectories in 2D space [{positions.shape[1]} episodes]')
        plt.gcf().dpi = 200
        plt.savefig(save_dir + '_multi_traj')
        # plt.show()

        return



def plot_one_episode(ep_data, save_dir=None, show=False):
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
    # info_history = ep_data['info']
    state_history = ep_data['obs']
    print(state_history.shape)
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
    plt.plot(tgrid, lin_vels[:, 0], color=cm.get_cmap('tab20')(0))
    plt.plot(tgrid, lin_vels[:, 1], color=cm.get_cmap('tab20')(2))
    plt.xlabel('Time steps')
    plt.ylabel('Velocity [m/s]')
    plt.legend(['x', 'y'], loc='best')
    plt.title('Velocity state history')
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + '_lin_vel')
    if show:
        plt.show()
    # °°°°°°°°°°°°°°°°°°°°°°°° plot angular speeds in time °°°°°°°°°°°°°°°°°°°°°°°°°
    ang_vel_z = state_history[:, 4:5]
    # plot angular speed (z coordinate)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, ang_vel_z, color=cm.get_cmap('tab20')(0))
    plt.xlabel('Time steps')
    plt.ylabel('Angular speed [rad/s]')
    plt.legend(['z'], loc='best')
    plt.title('Angular speed state history')
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + '_ang_vel')
    if show:
        plt.show()
    
    # °°°°°°°°°°°°°°°°°°°°°°°° plot heading cos, sin °°°°°°°°°°°°°°°°°°°°°°°°°
    headings = state_history[:, :2]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, headings[:, 0], color=cm.get_cmap('tab20')(0)) # cos
    plt.plot(tgrid, headings[:, 1], color=cm.get_cmap('tab20')(2)) # sin
    plt.xlabel('Time steps')
    plt.ylabel('Heading')
    plt.legend(['cos(${\\theta}$)', 'sin(${\\theta}$)'], loc='best')
    plt.title('Heading state history')
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + '_heading')
    if show:
        plt.show()

    # °°°°°°°°°°°°°°°°°°°°°°°° plot absolute heading angle °°°°°°°°°°°°°°°°°°°°°°°°°
    headings = state_history[:, :2]
    angles = np.arctan2(headings[:, 0], headings[:, 1])
    #angles = np.where(angles < 0, angles + np.pi, angles)
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, angles, color=cm.get_cmap('tab20')(0))
    plt.xlabel('Time steps')
    plt.ylabel('Angle [rad]')
    plt.legend(['${\\theta}$'], loc='best')
    plt.title('Angle state history')
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + '_angle')
    if show:
        plt.show()
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
    plt.xlabel('Time steps')
    if save_dir:
        fig.savefig(save_dir + '_actions')
    if show:
        plt.show()
    
        # °°°°°°°°°°°°°°°°°°°°°°°° plot actions histogram °°°°°°°°°°°°°°°°°°°°°°°°°
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = np.array(control_history)

    actions_df = pd.DataFrame(control_history, columns=[f'T{i+1}' for i in range(control_history.shape[1])])
    freq = actions_df.sum()
    plt.bar(freq.index, freq.values, color=cm.get_cmap('tab20')(0))
    plt.title('Number of thrusts in episode')
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + '_actions_hist')
    if show:
        plt.show()

    # °°°°°°°°°°°°°°°°°°°°°°°° plot rewards °°°°°°°°°°°°°°°°°°°°°°°°°
    if reward_history:
        fig_count += 1
        plt.figure(fig_count)
        plt.clf()
        plt.plot(tgrid, reward_history, color=cm.get_cmap('tab20')(0))
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
    plt.plot(tgrid, pos_error[:, 0], color=cm.get_cmap('tab20')(0))
    plt.plot(tgrid, pos_error[:, 1], color=cm.get_cmap('tab20')(2)) 
    plt.xlabel('Time steps')
    plt.ylabel('Position [m]')
    plt.legend(['x position', 'y position'], loc='best')
    plt.title('Planar position')
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + '_pos_error')
    if show:
        plt.show()

        # °°°°°°°°°°°°°°°°°°°°°°°° plot x, y position in plane °°°°°°°°°°°°°°°°°°°°°°°°°
    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    x, y = pos_error[:, 0], pos_error[:, 1]
    fig,ax = plt.subplots(figsize=(6,6))

     #Setting the limit of x and y direction to define which portion to zoom
    x1, x2, y1, y2 = -.07, .07, -.08, .08
    if (y[0] > 0 and x[0] > 0): 
        location = 4
    else:
        location = 2 if (y[0] < 0 and x[0] < 0) else 1
    axins = inset_axes(ax, width=1.5, height=1.25, loc=location)
    ax.plot(x, y, color=cm.get_cmap('tab20')(0))
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.plot(x, y)
    if save_dir:
        fig.savefig(save_dir + '_pos_xy_plane')
    if show:
        plt.show()

    # °°°°°°°°°°°°°°°°°°°°°°°° plot distance to target over time °°°°°°°°°°°°°°°°°°°°°°°°°
    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0),
             color=cm.get_cmap('tab20')(0))
    plt.xlabel('Time steps')
    plt.ylabel('Distance [m]')
    plt.legend(['abs dist'], loc='best')
    plt.title('Distance to target')
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + '_dist_to_target')
    if show:
        plt.show()

    # °°°°°°°°°°°°°°°°°°°°°°°° plot log-distance to target over time °°°°°°°°°°°°°°°°°°°°°°°°°
    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale('log')
    plt.plot(tgrid, np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0), 
             color=cm.get_cmap('tab20')(0))
    plt.xlabel('Time steps')
    plt.ylabel('Log distance [m]')
    plt.legend(['x-y dist'], loc='best')
    plt.title('Log distance to target')
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir + '_log_dist_to_target')
    if show:
        plt.show()
