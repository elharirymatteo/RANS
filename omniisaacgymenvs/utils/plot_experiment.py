__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"


import array
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import seaborn as sns
from matplotlib.collections import LineCollection


def plot_episode_data_virtual(
    ep_data: dict, save_dir: str, all_agents: bool = False
) -> None:
    """
    Plots the evaluation data for a single agent across a set of evaluation episodes.
    The following metrics are aggregated across all epsiodes:
    - distance to the goal
    - reward
    - velocities (angular and linear)
    - actions
    - trajectories: XY positions, no heading.

    Args:
    ep_data: dict: dictionary containing episode data
    save_dir: str: directory where to save the plots
    all_agents: bool: if True, plot average results over all agents, if False only the first agent is plotted
    """

    reward_history = ep_data["rews"]
    control_history = ep_data["act"]
    state_history = ep_data["obs"]

    # Overrides the user arg is there is only one episode
    all_agents = False if reward_history.shape[1] == 1 else all_agents

    fig_count = 0
    if all_agents:
        best_agent = np.argmax(reward_history.sum(axis=0))
        worst_agent = np.argmin(reward_history.sum(axis=0))
        rand_agent = np.random.choice(
            list(
                set(range(0, reward_history.shape[1])) - set([best_agent, worst_agent])
            )
        )
        print(
            "Best agent: ",
            best_agent,
            "| Worst agent: ",
            worst_agent,
            "| Random Agent",
            rand_agent,
        )
        # plot best and worst episodes data
        plot_one_episode(
            {k: np.array([v[best_agent] for v in vals]) for k, vals in ep_data.items()},
            save_dir + "best_ep/",
        )
        plot_one_episode(
            {
                k: np.array([v[worst_agent] for v in vals])
                for k, vals in ep_data.items()
            },
            save_dir + "worst_ep/",
        )
        plot_one_episode(
            {k: np.array([v[rand_agent] for v in vals]) for k, vals in ep_data.items()},
            save_dir + f"rand_ep_{rand_agent}/",
        )

        tgrid = np.linspace(0, len(reward_history), len(control_history))

        args = {
            "best_agent": best_agent,
            "worst_agent": worst_agent,
            "rand_agent": rand_agent,
            "fig_count": fig_count,
            "save_dir": save_dir,
            "reward_history": reward_history,
            "control_history": control_history,
            "state_history": state_history,
            "tgrid": tgrid,
        }

        shared_metrics = [plot_reward, plot_velocities, plot_actions_box_plot]
        task_metrics = []

        task_flag = state_history[0, 0, 5].astype(int)
        task_metrics = []
        if task_flag == 0:  # GoToXY
            task_metrics = [
                plot_trajectories_GoToXY,
                plot_distance_GoToXY,
                plot_all_distances_GoToXY,
            ]
        elif task_flag == 1:  # GoToPose
            task_metrics = [
                plot_trajectories_GoToXY,
                plot_distance_GoToPose,
                plot_all_distances_GoToPose,
            ]
        elif task_flag == 2:  # TrackXYVelocity
            task_metrics = [
                plot_distance_TrackXYVelocity,
                plot_all_distances_TrackXYVelocity,
            ]
        elif task_flag == 3:  # TrackXYOVelocity
            task_metrics = [
                plot_distance_TrackXYOVelocity,
                plot_all_distances_TrackXYOVelocity,
            ]
        else:
            task_metrics = []
        metrics = shared_metrics + task_metrics

        for metric in metrics:
            fig_count = metric(**args)
            args["fig_count"] = fig_count

    else:
        fig_count = plot_one_episode(
            {k: np.array([v[0] for v in vals]) for k, vals in ep_data.items()},
            save_dir + "first_ep/",
        )


def plot_distance_GoToXY(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    best_agent: int,
    worst_agent: int,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot mean, std_dev, best and worst distance over all episodes."""

    all_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)

    fig_count += 1
    fig, ax = plt.subplots()
    ax.plot(
        tgrid,
        all_distances.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        all_distances.mean(axis=1) - all_distances.std(axis=1),
        all_distances.mean(axis=1) + all_distances.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        all_distances.mean(axis=1) - 2 * all_distances.std(axis=1),
        all_distances.mean(axis=1) + 2 * all_distances.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        all_distances[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        all_distances[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m]")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(f"Mean, best and worst distances over {all_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_position_distances")
    return fig_count


def plot_all_distances_GoToXY(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot all distances over all episodes."""

    all_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)

    fig_count += 1
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab20")
    for j in range(all_distances.shape[1]):
        ax.plot(
            tgrid, all_distances[:, j], alpha=1.0, color=cmap(j % cmap.N), linewidth=1.0
        )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m]")
    plt.title(f"All distances over {all_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "all_position_distances")
    return fig_count


def plot_distance_GoToPose(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    best_agent: int,
    worst_agent: int,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot mean, std_dev, best and worst distance over all episodes."""

    all_position_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)
    shape = state_history.shape[:-1]
    all_heading_distances = np.arctan2(
        state_history[:, :, 9].flatten(), state_history[:, :, 8].flatten()
    )
    all_heading_distances = all_heading_distances.reshape(shape)

    fig_count += 1
    fig, ax = plt.subplots()
    ax.plot(
        tgrid,
        all_position_distances.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        all_position_distances.mean(axis=1) - all_position_distances.std(axis=1),
        all_position_distances.mean(axis=1) + all_position_distances.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        all_position_distances.mean(axis=1) - 2 * all_position_distances.std(axis=1),
        all_position_distances.mean(axis=1) + 2 * all_position_distances.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        all_position_distances[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        all_position_distances[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m]")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(
        f"Mean, best and worst distances over {all_position_distances.shape[1]} episodes"
    )
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_position_distances")

    fig_count += 1
    fig, ax = plt.subplots()
    ax.plot(
        tgrid,
        all_heading_distances.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        all_heading_distances.mean(axis=1) - all_heading_distances.std(axis=1),
        all_heading_distances.mean(axis=1) + all_heading_distances.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        all_heading_distances.mean(axis=1) - 2 * all_heading_distances.std(axis=1),
        all_heading_distances.mean(axis=1) + 2 * all_heading_distances.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        all_heading_distances[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        all_heading_distances[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [rad]")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(
        f"Mean, best and worst distances over {all_heading_distances.shape[1]} episodes"
    )
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_heading_distances")
    return fig_count


def plot_all_distances_GoToPose(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot all distances over all episodes."""

    all_position_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)
    shape = state_history.shape[:-1]
    all_heading_distances = np.arctan2(
        state_history[:, :, 9].flatten(), state_history[:, :, 8].flatten()
    )
    all_heading_distances = all_heading_distances.reshape(shape)

    fig_count += 1
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab20")
    for j in range(all_position_distances.shape[1]):
        ax.plot(
            tgrid,
            all_position_distances[:, j],
            alpha=1.0,
            color=cmap(j % cmap.N),
            linewidth=1.0,
        )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m]")
    plt.title(f"All distances over {all_position_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "all_position_distances")

    fig_count += 1
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab20")
    for j in range(all_heading_distances.shape[1]):
        ax.plot(
            tgrid,
            all_heading_distances[:, j],
            alpha=1.0,
            color=cmap(j % cmap.N),
            linewidth=1.0,
        )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [rad]")
    plt.title(f"All distances over {all_heading_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "all_heading_distances")
    return fig_count


def plot_distance_TrackXYVelocity(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    best_agent: int,
    worst_agent: int,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot mean, std_dev, best and worst distance over all episodes."""

    all_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)

    fig_count += 1
    fig, ax = plt.subplots()
    ax.plot(
        tgrid,
        all_distances.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        all_distances.mean(axis=1) - all_distances.std(axis=1),
        all_distances.mean(axis=1) + all_distances.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        all_distances.mean(axis=1) - 2 * all_distances.std(axis=1),
        all_distances.mean(axis=1) + 2 * all_distances.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        all_distances[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        all_distances[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m/s]")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(f"Mean, best and worst distances over {all_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_velocity_distances")
    return fig_count


def plot_all_distances_TrackXYVelocity(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot all distances over all episodes."""

    all_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)

    fig_count += 1
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab20")
    for j in range(all_distances.shape[1]):
        ax.plot(
            tgrid, all_distances[:, j], alpha=1.0, color=cmap(j % cmap.N), linewidth=1.0
        )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m/s]")
    plt.title(f"All distances over {all_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "all_velocity_distances")
    return fig_count


def plot_distance_TrackXYOVelocity(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    best_agent: int,
    worst_agent: int,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot mean, std_dev, best and worst distance over all episodes."""

    all_xy_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)
    all_omega_distances = np.linalg.norm(state_history[:, :, 8], axis=2)

    fig_count += 1
    fig, ax = plt.subplots()
    ax.plot(
        tgrid,
        all_xy_distances.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        all_xy_distances.mean(axis=1) - all_xy_distances.std(axis=1),
        all_xy_distances.mean(axis=1) + all_xy_distances.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        all_xy_distances.mean(axis=1) - 2 * all_xy_distances.std(axis=1),
        all_xy_distances.mean(axis=1) + 2 * all_xy_distances.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        all_xy_distances[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        all_xy_distances[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m/s]")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(
        f"Mean, best and worst distances over {all_xy_distances.shape[1]} episodes"
    )
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_velocity_distances")

    fig_count += 1
    fig, ax = plt.subplots()
    ax.plot(
        tgrid,
        all_omega_distances.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        all_omega_distances.mean(axis=1) - all_omega_distances.std(axis=1),
        all_omega_distances.mean(axis=1) + all_omega_distances.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        all_omega_distances.mean(axis=1) - 2 * all_omega_distances.std(axis=1),
        all_omega_distances.mean(axis=1) + 2 * all_omega_distances.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        all_omega_distances[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        all_omega_distances[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m/s]")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(
        f"Mean, best and worst distances over {all_omega_distances.shape[1]} episodes"
    )
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_velocity_distances")
    return fig_count


def plot_all_distances_TrackXYOVelocity(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot all distances over all episodes."""

    all_xy_distances = np.linalg.norm(state_history[:, :, 6:8], axis=2)
    all_omega_distances = np.linalg.norm(state_history[:, :, 8], axis=2)

    fig_count += 1
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab20")
    for j in range(all_xy_distances.shape[1]):
        ax.plot(
            tgrid,
            all_xy_distances[:, j],
            alpha=1.0,
            color=cmap(j % cmap.N),
            linewidth=1.0,
        )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m/s]")
    plt.title(f"All distances over {all_xy_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "all_velocity_distances")

    fig_count += 1
    fig, ax = plt.subplots()
    cmap = cm.get_cmap("tab20")
    for j in range(all_omega_distances.shape[1]):
        ax.plot(
            tgrid,
            all_omega_distances[:, j],
            alpha=1.0,
            color=cmap(j % cmap.N),
            linewidth=1.0,
        )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [rad/s]")
    plt.title(f"All distances over {all_omega_distances.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "all_velocity_distances")
    return fig_count


def plot_reward(
    reward_history: np.ndarray,
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    best_agent: int,
    worst_agent: int,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot mean, std_dev, best and worst reward over all episodes."""

    fig_count += 1
    fig, ax = plt.subplots()

    ax.plot(
        tgrid,
        reward_history.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        reward_history.mean(axis=1) - reward_history.std(axis=1),
        reward_history.mean(axis=1) + reward_history.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        reward_history.mean(axis=1) - 2 * reward_history.std(axis=1),
        reward_history.mean(axis=1) + 2 * reward_history.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        reward_history[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        reward_history[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Reward")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(f"Mean, best and worst reward over {state_history.shape[1]} episodes")
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_rewards")
    return fig_count


def plot_velocities(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    best_agent: int,
    worst_agent: int,
    fig_count: int,
    **kwargs,
) -> int:
    """
    Plot mean, std_dev, best and worst velocities over all episodes."""

    fig_count += 1
    fig, ax = plt.subplots()
    ang_vel_z = state_history[:, :, 4:5][:, :, 0]  # getting rid of the extra dimension

    ax.plot(
        tgrid,
        ang_vel_z.mean(axis=1),
        alpha=0.5,
        color="blue",
        label="mean_dist",
        linewidth=1.5,
    )
    ax.fill_between(
        tgrid,
        ang_vel_z.mean(axis=1) - ang_vel_z.std(axis=1),
        ang_vel_z.mean(axis=1) + ang_vel_z.std(axis=1),
        color="blue",
        alpha=0.4,
    )
    ax.fill_between(
        tgrid,
        ang_vel_z.mean(axis=1) - 2 * ang_vel_z.std(axis=1),
        ang_vel_z.mean(axis=1) + 2 * ang_vel_z.std(axis=1),
        color="blue",
        alpha=0.2,
    )
    ax.plot(
        tgrid,
        ang_vel_z[:, best_agent],
        alpha=0.5,
        color="green",
        label="best",
        linewidth=1.5,
    )
    ax.plot(
        tgrid,
        ang_vel_z[:, worst_agent],
        alpha=0.5,
        color="red",
        label="worst",
        linewidth=1.5,
    )
    plt.xlabel("Time steps")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(["mean", "best", "worst", "1-std", "2-std"], loc="best")
    plt.title(
        f"Angular speed of mean, best and worst agents {ang_vel_z.shape[1]} episodes"
    )
    plt.grid()
    plt.savefig(save_dir + "mean_best_worst_ang_velocities")
    return fig_count


def plot_actions_histogram(
    control_history: np.ndarray, save_dir: str, fig_count: int, **kwargs
) -> int:
    """
    Plot mean number of thrusts over all episodes."""

    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = control_history.reshape(
        (control_history.shape[1], control_history.shape[0], control_history.shape[2])
    )
    control_history = np.array([c for c in control_history])

    freq = pd.DataFrame(
        data=np.array(
            [control_history[i].sum(axis=0) for i in range(control_history.shape[0])]
        ),
        columns=[f"T{i+1}" for i in range(control_history.shape[2])],
    )
    mean_freq = freq.mean()
    plt.bar(mean_freq.index, mean_freq.values)
    plt.title(f"Mean number of thrusts in {control_history.shape[0]} episodes")
    plt.savefig(save_dir + "mean_actions_histogram")
    return fig_count


def plot_actions_box_plot(
    control_history: np.ndarray, save_dir: str, fig_count: int, **kwargs
) -> int:
    """
    Plot box plot of actions over all episodes."""

    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = control_history.reshape(
        (control_history.shape[1], control_history.shape[0], control_history.shape[2])
    )
    control_history = np.array([c for c in control_history])

    freq = pd.DataFrame(
        data=np.array(
            [control_history[i].sum(axis=0) for i in range(control_history.shape[0])]
        ),
        columns=[f"T{i+1}" for i in range(control_history.shape[2])],
    )
    sns.boxplot(data=freq, orient="h")
    plt.title(f"Mean number of thrusts in {control_history.shape[0]} episodes")
    plt.savefig(save_dir + "actions_boxplot")
    return fig_count


def plot_trajectories_GoToXY(
    state_history: np.ndarray, save_dir: str, fig_count: int, **kwargs
) -> int:
    """
    Plot trajectories of all agents in 2D space."""

    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    positions = state_history[:, :, 6:8]

    cmap = cm.get_cmap("tab20")
    for j in range(positions.shape[1]):
        col = cmap(
            j % cmap.N
        )  # Select a color from the colormap based on the current index
        plt.plot(
            positions[:, j, 0], positions[:, j, 1], color=col, alpha=1.0, linewidth=0.75
        )
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")

    plt.grid(alpha=0.3)
    plt.title(f"Trajectories in 2D space [{positions.shape[1]} episodes]")
    plt.gcf().dpi = 200
    plt.savefig(save_dir + "multi_trajectories")
    return fig_count


def plot_one_episode(
    ep_data: dict,
    save_dir: str = None,
    show: bool = False,
    debug: bool = False,
    fig_count: int = 0,
) -> None:
    """
    Plot episode metrics for a single agent.

    ep_data: dictionary containing episode data
    save_dir: directory where to save the plots
    all_agents: if True, plot average results over all agents, if False only the first agent is plotted.
    """
    os.makedirs(save_dir, exist_ok=True)

    control_history = ep_data["act"]
    reward_history = ep_data["rews"]
    state_history = ep_data["obs"]

    # save data to csv file
    pd.DataFrame.to_csv(pd.DataFrame(control_history), save_dir + "actions.csv")

    shared_metrics = [
        plot_single_linear_vel,
        plot_single_angular_vel,
        plot_single_absolute_heading,
        plot_single_rewards,
        plot_single_action_histogram,
        plot_single_actions,
    ]

    if debug:
        debug_metrics = [plot_single_heading_cos_sin]
    else:
        debug_metrics = []

    # setting the right task_data labels based on the task fla.
    task_flag = state_history[0, 5].astype(int)

    task_metrics = []
    if task_flag == 0:  # GoToXY
        task_data_label = ["error_x", "error_y"]
        task_metrics = [
            plot_single_xy_position,
            plot_single_xy_position_error,
            plot_single_GoToXY_distance_to_target,
            plot_single_GoToXY_log_distance_to_target,
        ]
    elif task_flag == 1:  # GoToPose
        task_data_label = [
            "error_x",
            "error_y",
            "cos_error_heading",
            "sin_error_heading",
        ]
        task_metrics = [
            plot_single_xy_position,
            plot_single_xy_pose,
            plot_single_xy_position_error,
            plot_single_heading_error,
            plot_single_xy_position_heading,
            plot_single_GoToPose_distance_to_target,
            plot_single_GoToPose_log_distance_to_target,
        ]
    elif task_flag == 2:  # TrackXYVelocity
        task_data_label = ["error_vx", "error_vy"]
        task_metrics = [
            plot_single_TrackXYVelocity_distance_to_target,
            plot_single_TrackXYVelocity_log_distance_to_target,
        ]
    elif task_flag == 3:  # TrackXYOVelocity
        task_data_label = ["error_vx", "error_vy", "error_omega"]
        task_metrics = [
            plot_single_TrackXYOVelocity_distance_to_target,
            plot_single_TrackXYOVelocity_log_distance_to_target,
        ]
    else:
        task_data_label = []
        task_metrics = []

    # Generate plots
    metrics = shared_metrics + task_metrics + debug_metrics
    tgrid = np.linspace(0, len(control_history), len(control_history))
    args = {
        "state_history": state_history,
        "control_history": control_history,
        "reward_history": reward_history,
        "save_dir": save_dir,
        "fig_count": fig_count,
        "show": show,
        "tgrid": tgrid,
    }

    for metric in metrics:
        fig_count = metric(**args)
        args["fig_count"] = fig_count

    df_cols = [
        "cos_theta",
        "sin_theta",
        "lin_vel_x",
        "lin_vel_y",
        "ang_vel_z",
        "task_flag",
    ] + task_data_label
    pd.DataFrame.to_csv(
        pd.DataFrame(state_history[:, : len(df_cols)], columns=df_cols),
        save_dir + "states_episode.csv",
    )

    fig_count = 0


def plot_single_linear_vel(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot linear velocity of a single agent."""

    lin_vels = state_history[:, 2:4]
    # plot linear velocity
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, lin_vels[:, 0], color=cm.get_cmap("tab20")(0))
    plt.plot(tgrid, lin_vels[:, 1], color=cm.get_cmap("tab20")(2))
    plt.xlabel("Time steps")
    plt.ylabel("Velocity [m/s]")
    plt.legend(["x", "y"], loc="best")
    plt.title("Velocity state history")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_linear_velocity")
    if show:
        plt.show()
    return fig_count


def plot_single_angular_vel(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot angular velocity of a single agent."""

    ang_vel_z = state_history[:, 4:5]
    # plot angular speed (z coordinate)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, ang_vel_z, color=cm.get_cmap("tab20")(0))
    plt.xlabel("Time steps")
    plt.ylabel("Angular speed [rad/s]")
    plt.legend(["z"], loc="best")
    plt.title("Angular speed state history")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_angular_velocity")
    if show:
        plt.show()
    return fig_count


def plot_single_heading_cos_sin(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot heading of a single agent with cos and sin representation."""

    headings = state_history[:, :2]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, headings[:, 0], color=cm.get_cmap("tab20")(0))  # cos
    plt.plot(tgrid, headings[:, 1], color=cm.get_cmap("tab20")(2))  # sin
    plt.xlabel("Time steps")
    plt.ylabel("Heading")
    plt.legend(["cos(${\\theta}$)", "sin(${\\theta}$)"], loc="best")
    plt.title("Heading state history")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_heading_cos_sin")
    if show:
        plt.show()
    return fig_count


def plot_single_absolute_heading(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot heading of a single agent."""

    headings = state_history[:, :2]
    angles = np.arctan2(headings[:, 1], headings[:, 0])
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, angles, color=cm.get_cmap("tab20")(0))
    plt.xlabel("Time steps")
    plt.ylabel("Angle [rad]")
    plt.legend(["${\\theta}$"], loc="best")
    plt.title("Angle state history")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_heading")
    if show:
        plt.show()
    return fig_count


def plot_single_actions(
    control_history: np.ndarray, save_dir: str, fig_count: int, show: bool, **kwargs
) -> int:
    """
    Plot actions of a single agent."""

    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history_df = pd.DataFrame(data=control_history)

    fig, axes = plt.subplots(
        len(control_history_df.columns), 1, sharex=True, figsize=(8, 6)
    )
    # Select subset of colors from a colormap
    colormap = cm.get_cmap("tab20")
    num_colors = len(control_history_df.columns)
    colors = [colormap(i) for i in range(0, num_colors * 2, 2)]
    for i, column in enumerate(control_history_df.columns):
        control_history_df[column].plot(ax=axes[i], color=colors[i])
        axes[i].set_ylabel(f"T{column}")
    plt.xlabel("Time steps")
    if save_dir:
        fig.savefig(save_dir + "single_actions")
    if show:
        plt.show()
    return fig_count


def plot_single_action_histogram(
    control_history: np.ndarray, save_dir: str, fig_count: int, show: bool, **kwargs
) -> int:
    """
    Plot histogram of actions of a single agent."""

    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    control_history = np.array(control_history)

    actions_df = pd.DataFrame(
        control_history, columns=[f"T{i+1}" for i in range(control_history.shape[1])]
    )
    freq = actions_df.sum()
    plt.bar(freq.index, freq.values, color=cm.get_cmap("tab20")(0))
    plt.title("Number of thrusts in episode")
    plt.tight_layout()
    if save_dir:
        plt.savefig(save_dir + "single_actions_hist")
    if show:
        plt.show()
    return fig_count


def plot_single_rewards(
    reward_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot rewards of a single agent."""

    if any(reward_history):
        fig_count += 1
        plt.figure(fig_count)
        plt.clf()
        plt.plot(tgrid, reward_history, color=cm.get_cmap("tab20")(0))
        plt.xlabel("Time steps")
        plt.ylabel("Reward")
        plt.legend(["reward"], loc="best")
        plt.title("Reward history")
        plt.grid()
        if save_dir:
            plt.savefig(save_dir + "single_reward")
        if show:
            plt.show()
    return fig_count


def plot_single_xy_position_error(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot position error of a single agent."""

    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, pos_error[:, 0], color=cm.get_cmap("tab20")(0))
    plt.plot(tgrid, pos_error[:, 1], color=cm.get_cmap("tab20")(2))
    plt.xlabel("Time steps")
    plt.ylabel("Position [m]")
    plt.legend(["x position", "y position"], loc="best")
    plt.title("Position Error")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_xy_position_error")
    if show:
        plt.show()
    return fig_count


def plot_single_heading_error(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot heading error of a single agent."""

    heading_error = state_history[:, 8:]
    heading_error = np.arctan2(heading_error[:, 1], heading_error[:, 0])
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, heading_error, color=cm.get_cmap("tab20")(0))
    plt.xlabel("Time steps")
    plt.ylabel("Heading [rad]")
    plt.title("Heading error")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_heading_heading_error")
    if show:
        plt.show()
    return fig_count


def plot_single_xy_position(
    state_history: np.ndarray, save_dir: str, fig_count: int, show: bool, **kwargs
) -> int:
    """
    Plot position of a single agent."""

    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    # Set aspect ratio to be equal
    plt.gca().set_aspect("equal", adjustable="box")
    x, y = pos_error[:, 0], pos_error[:, 1]
    fig, ax = plt.subplots(figsize=(6, 6))

    # Setting the limit of x and y direction to define which portion to zoom
    x1, x2, y1, y2 = -0.07, 0.07, -0.08, 0.08
    if y[0] > 0 and x[0] > 0:
        location = 4
    else:
        location = 2 if (y[0] < 0 and x[0] < 0) else 1
    axins = inset_axes(ax, width=1.5, height=1.25, loc=location)
    ax.plot(x, y, color=cm.get_cmap("tab20")(0))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.plot(x, y)
    if save_dir:
        fig.savefig(save_dir + "single_xy_trajectory")
    if show:
        plt.show()
    return fig_count


def plot_single_xy_pose(
    state_history: np.ndarray, save_dir: str, fig_count: int, show: bool, **kwargs
) -> int:
    """
    Plot position of a single agent."""

    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    # Set aspect ratio to be equal
    plt.gca().set_aspect("equal", adjustable="box")

    # Get the heading error values
    heading_error = state_history[:, 8:]
    heading_error = np.abs(np.arctan2(heading_error[:, 1], heading_error[:, 0]))
    x, y = pos_error[:, 0], pos_error[:, 1]
    segments = [
        np.column_stack([x[i : i + 2], y[i : i + 2]]) for i in range(len(x) - 1)
    ]

    fig, ax = plt.subplots(figsize=(7, 6))
    # make sure that the plot won't be limited between 0 and 1, ensuring the limits derive from the x, y coordinates plus a margin
    margin = 0.08
    ax.set_xlim(min(x) - margin, max(x) + margin)
    ax.set_ylim(min(y) - margin, max(y) + margin)
    lc = LineCollection(segments, cmap="jet", array=heading_error)
    line = ax.add_collection(lc)
    plt.colorbar(line, label="heading error [rad]")
    # ax.plot(x, y, color=cm.get_cmap('tab20')(0))
    plt.grid(alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")

    plt.grid(alpha=0.3)
    if save_dir:
        fig.savefig(save_dir + "single_pose_trajectory")
    if show:
        plt.show()
    return fig_count


def plot_single_xy_position_heading(
    state_history: np.ndarray, save_dir: str, fig_count: int, show: bool, **kwargs
) -> int:
    """
    Plot position of a single agent."""

    pos_error = state_history[:, 6:8]
    heading = state_history[:, :2]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    # Set aspect ratio to be equal
    plt.gca().set_aspect("equal", adjustable="box")
    x, y = pos_error[:, 0], pos_error[:, 1]
    c, s = heading[:, 0], heading[:, 1]
    fig, ax = plt.subplots(figsize=(6, 6))

    # Setting the limit of x and y direction to define which portion to zoom
    x1, x2, y1, y2 = -0.07, 0.07, -0.08, 0.08
    if y[0] > 0 and x[0] > 0:
        location = 4
    else:
        location = 2 if (y[0] < 0 and x[0] < 0) else 1
    axins = inset_axes(ax, width=1.5, height=1.25, loc=location)
    ax.plot(x, y, color=cm.get_cmap("tab20")(0))
    ax.quiver(x[::10], y[::10], s[::10], c[::10], color=cm.get_cmap("tab20")(0))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    axins.plot(x, y)
    if save_dir:
        fig.savefig(save_dir + "single_xy_trajectory_with_heading")
    if show:
        plt.show()
    return fig_count


def plot_single_GoToXY_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot distance to target of a single agent."""

    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m]")
    plt.legend(["abs dist"], loc="best")
    plt.title("Distance to target")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_distance_to_position_target")
    if show:
        plt.show()
    return fig_count


def plot_single_GoToXY_log_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot log distance to target of a single agent."""

    pos_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale("log")
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Log distance [m]")
    plt.legend(["x-y dist"], loc="best")
    plt.title("Log distance to target")
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir + "single_log_distance_to_position_target")
    if show:
        plt.show()
    return fig_count


def plot_single_GoToPose_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot distance to target of a single agent."""

    pos_error = state_history[:, 6:8]
    heading_error = state_history[:, 8:]
    heading_error = np.arctan2(heading_error[:, 1], heading_error[:, 0])
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m]")
    plt.legend(["abs dist"], loc="best")
    plt.title("Distance to target")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_distance_to_position_target")
    if show:
        plt.show()
    # plot heading
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, np.abs(heading_error), color=cm.get_cmap("tab20")(0))
    plt.xlabel("Time steps")
    plt.ylabel("Heading [rad]")
    plt.legend(["abs dist"], loc="best")
    plt.title("Distance to target")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_distance_to_heading_target")
    if show:
        plt.show()
    return fig_count


def plot_single_GoToPose_log_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot log distance to target of a single agent."""

    pos_error = state_history[:, 6:8]
    heading_error = state_history[:, 8:]
    heading_error = np.arctan2(heading_error[:, 1], heading_error[:, 0])
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale("log")
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([pos_error[:, 0], pos_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Log distance [m]")
    plt.legend(["x-y dist"], loc="best")
    plt.title("Log distance to target")
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir + "single_log_distance_to_position_target")
    if show:
        plt.show()
    # plot heading
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale("log")
    plt.plot(tgrid, np.abs(heading_error), color=cm.get_cmap("tab20")(0))
    plt.xlabel("Time steps")
    plt.ylabel("Log distance [rad]")
    plt.legend(["x-y dist"], loc="best")
    plt.title("Log distance to target")
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir + "single_log_distance_to_heading_target")
    if show:
        plt.show()
    return fig_count


def plot_single_TrackXYVelocity_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot distance to target of a single agent."""

    vel_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([vel_error[:, 0], vel_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m/s]")
    plt.legend(["abs dist"], loc="best")
    plt.title("Distance to target")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_distance_to_velocity_target")
    if show:
        plt.show()
    return fig_count


def plot_single_TrackXYVelocity_log_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot log distance to target of a single agent."""

    vel_error = state_history[:, 6:8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale("log")
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([vel_error[:, 0], vel_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Log distance [m/s]")
    plt.legend(["x-y dist"], loc="best")
    plt.title("Log distance to target")
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir + "single_log_distance_to_velocity_target")
    if show:
        plt.show()
    return fig_count


def plot_single_TrackXYOVelocity_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot distance to target of a single agent."""

    vel_error = state_history[:, 6:8]
    omega_error = state_history[:, 8]
    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([vel_error[:, 0], vel_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Distance [m/s]")
    plt.legend(["abs dist"], loc="best")
    plt.title("Distance to target")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_distance_to_linear_velocity_target")
    if show:
        plt.show()

    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.plot(tgrid, np.abs(omega_error), color=cm.get_cmap("tab20")(0))
    plt.xlabel("Time steps")
    plt.ylabel("Distance [rad/s]")
    plt.legend(["abs dist"], loc="best")
    plt.title("Distance to target")
    plt.grid()
    if save_dir:
        plt.savefig(save_dir + "single_distance_to_angular_velocity_target")
    if show:
        plt.show()
    return fig_count


def plot_single_TrackXYOVelocity_log_distance_to_target(
    state_history: np.ndarray,
    tgrid: np.ndarray,
    save_dir: str,
    fig_count: int,
    show: bool,
    **kwargs,
) -> int:
    """
    Plot log distance to target of a single agent."""

    vel_error = state_history[:, 6:8]
    omega_error = state_history[:, 8]

    # plot position (x, y coordinates)
    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale("log")
    plt.plot(
        tgrid,
        np.linalg.norm(np.array([vel_error[:, 0], vel_error[:, 1]]), axis=0),
        color=cm.get_cmap("tab20")(0),
    )
    plt.xlabel("Time steps")
    plt.ylabel("Log distance [m/s]")
    plt.legend(["x-y dist"], loc="best")
    plt.title("Log distance to target")
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir + "single_log_distance_to_linear_velocity_target")
    if show:
        plt.show()

    fig_count += 1
    plt.figure(fig_count)
    plt.clf()
    plt.yscale("log")
    plt.plot(tgrid, np.abs(omega_error), color=cm.get_cmap("tab20")(0))
    plt.xlabel("Time steps")
    plt.ylabel("Log distance [rad/s]")
    plt.legend(["x-y dist"], loc="best")
    plt.title("Log distance to target")
    plt.grid(True)
    if save_dir:
        plt.savefig(save_dir + "single_log_distance_to_angular_velocity_target")
    if show:
        plt.show()
    return fig_count
