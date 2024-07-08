import wandb
import sys
import matplotlib.pyplot as plt
import os
import re
import numpy as np
from collections import defaultdict

def login_to_wandb():
    wandb.login()

def get_project(entity, project_name):
    api = wandb.Api()
    try:
        return api.runs(f"{entity}/{project_name}")
    except wandb.errors.CommError:
        print("Error: Project not found. Please check the entity and project name.")
        sys.exit(1)

def categorize_runs(runs):
    categorized = defaultdict(lambda: defaultdict(list))
    for run in runs:
        match = re.match(r'(\w+)_(\w+)_seed\d+.*', run.name)
        if match:
            robot, task = match.groups()
            categorized[robot][task].append(run)
    return categorized

def categorize_runs_by_robot(runs):
    categorized = defaultdict(list)
    for run in runs:
        match = re.match(r'(\w+)_(\w+)_seed\d+.*', run.name)
        if match:
            robot, task = match.groups()
            categorized[robot].append(run)
    return categorized

def list_categorized_runs(categorized_runs):
    for robot, tasks in categorized_runs.items():
        print(f"\nRobot: {robot}")
        for task, runs in tasks.items():
            print(f"  Task: {task}")
            for run in runs:
                print(f"    {run.name}")

def extract_seed(input_string):
    match = re.search(r'_(seed\d+)_', input_string)
    if match:
        return match.group(1)
    else:
        return "unknown_seed"

def list_available_metrics(run):
    metrics = sorted(run.summary._json_dict.keys())
    metric_names = {index + 1: metric for index, metric in enumerate(metrics)}
    print("\nAvailable metrics:")
    for index, metric in metric_names.items():
        print(f"  {index}. {metric}")
    return metric_names

def plot_metric_for_runs(runs, metric_name, save_path, plot_type, x_axis='global_step'):
    all_histories = []
    
    for run in runs:
        run_name = run.name
        history = run.history(keys=[x_axis, metric_name])
        if not history.empty:
            steps = history[x_axis].values
            values = history[metric_name].values
            all_histories.append((steps, values))
            if plot_type == "all":
                seed = extract_seed(run_name)
                plt.plot(steps, values, label=f"{seed}")

    if plot_type == "average" and all_histories:
        min_length = min(len(values) for _, values in all_histories)
        all_steps = all_histories[0][0][:min_length]
        all_values = np.array([values[:min_length] for _, values in all_histories])

        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        plt.plot(all_steps, mean_values, label="Mean")
        plt.fill_between(all_steps, mean_values - std_values, mean_values + std_values, alpha=0.3, label="Std Dev")

    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over time for runs')
    plt.legend(loc='best')

    if save_path:
        plot_filename = os.path.join(save_path, f"{metric_name.replace('/', '_')}_{plot_type}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved as {plot_filename}")
    else:
        plt.show()

def plot_average_metric_for_task(runs, task, metric_name, x_axis='global_step', save_path=None):
    categorized_runs = categorize_runs_by_robot(runs)
    all_histories = defaultdict(list)

    for robot, robot_runs in categorized_runs.items():
        for run in robot_runs:
            if task in run.name:
                history = run.history(keys=[x_axis, metric_name])
                if not history.empty:
                    steps = history[x_axis].values
                    values = history[metric_name].values
                    all_histories[robot].append((steps, values))

    plt.figure(figsize=(10, 6))

    for robot, histories in all_histories.items():
        if histories:
            min_length = min(len(values) for _, values in histories)
            all_steps = histories[0][0][:min_length]
            all_values = np.array([values[:min_length] for _, values in histories])

            mean_values = np.mean(all_values, axis=0)
            std_values = np.std(all_values, axis=0)

            plt.plot(all_steps, mean_values, label=f"{robot}")
            plt.fill_between(all_steps, mean_values - std_values, mean_values + std_values, alpha=0.3)

    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over time for task "{task}" across all robots')
    plt.legend()

    if save_path:
        plot_filename = f"{save_path}/{metric_name.replace('/', '_')}_all_robots.png"
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved as {plot_filename}")
    else:
        plt.show()

def plot_average_metric_for_all_tasks(runs, metric_name, save_path=None):
    categorized = defaultdict(lambda: defaultdict(list))
    for run in runs:
        match = re.match(r'(\w+)_(\w+)_seed\d+.*', run.name)
        if match:
            robot, task = match.groups()
            categorized[task][robot].append(run)

    for task, robots in categorized.items():
        task_runs = [run for robot_runs in robots.values() for run in robot_runs]
        plot_average_metric_for_task(task_runs, task, metric_name, save_path=save_path+'/'+task if save_path else None)


def main():
    login_to_wandb()

    default_entity = "spacer-rl"  # Replace with your default entity
    default_project_name = "iclr_benchmark_10s"  # Replace with your default project name

    use_defaults = input("Do you want to use the default entity and project? (yes/no): ").strip().lower()
    if use_defaults == "yes" or use_defaults == "y":
        entity = default_entity
        project_name = default_project_name
    else:
        entity = input("Enter the wandb entity: ").strip()
        project_name = input("Enter the wandb project name: ").strip()

    runs = get_project(entity, project_name)
    categorized_runs = categorize_runs(runs)
    list_categorized_runs(categorized_runs)

    while True:
        print("\nOptions:")
        print("1. Select a robot and task pair to plot metrics")
        print("2. Plot average metric for all robots and tasks")
        print("3. Interactive session (original functionality)")
        print("4. Print the average metric for a task across all robots")
        print("5. Exit")
        choice = input("Choose an option (1, 2, 3, 4 or 5): ").strip()

        if choice == "1":
            robot = input("Enter the robot name: ").strip()
            task = input("Enter the task name: ").strip()
            if robot in categorized_runs and task in categorized_runs[robot]:
                selected_runs = categorized_runs[robot][task]
                metric_names = list_available_metrics(selected_runs[0])
                metric_number = int(input("Enter the metric number to plot: ").strip())
                metric_name = metric_names.get(metric_number)
                if metric_name:
                    plot_type = input("Do you want to plot all runs or average with std dev? (all/average): ").strip().lower()
                    save_choice = input("Do you want to save the plot? (yes/no): ").strip().lower()
                    if save_choice == "yes" or save_choice == "y":
                        save_path = os.path.join('wandb_data', f"{robot}/{task}")
                        os.makedirs(save_path, exist_ok=True)
                        plot_metric_for_runs(selected_runs, metric_name, save_path, plot_type)
                    else:
                        plot_metric_for_runs(selected_runs, metric_name, None, plot_type)
                else:
                    print("Metric number not found.")
            else:
                print("Robot or task not found.")
        elif choice == "2":
            metric_name = input("Enter the metric name to plot: ").strip()
            save_choice = input("Do you want to save the plots? (yes/no): ").strip().lower()
            if save_choice == "yes" or save_choice == "y":
                save_path = 'wandb_data'
                os.makedirs(save_path, exist_ok=True)
                plot_average_metric_for_all_tasks(runs, metric_name, save_path)
            else:
                plot_average_metric_for_all_tasks(runs, metric_name)
        elif choice == "3":
            keyword = input("Enter the keyword to filter runs: ").strip()
            filtered_runs = [run for run in runs if keyword in run.name or keyword in str(run.config)]
            if filtered_runs:
                metric_names = list_available_metrics(filtered_runs[0])
                metric_number = int(input("Enter the metric number to plot: ").strip())
                metric_name = metric_names.get(metric_number)
                if metric_name:
                    plot_type = input("Do you want to plot all runs or average with std dev? (all/average): ").strip().lower()
                    save_choice = input("Do you want to save the plot? (yes/no): ").strip().lower()
                    if save_choice == "yes" or save_choice == "y":
                        save_path = os.path.join('wandb_data', keyword)
                        if save_path:
                            os.makedirs(save_path, exist_ok=True)
                        plot_metric_for_runs(filtered_runs, metric_name, save_path, plot_type, keyword)
                    else:
                        plot_metric_for_runs(filtered_runs, metric_name, None, plot_type, keyword)
                else:
                    print("Metric number not found.")
            else:
                print("No runs found with the specified keyword.")       
        elif choice == "4":
            task = input("Enter the task name: ").strip()
            metric_names = list_available_metrics(runs[0])
            metric_number = int(input("Enter the metric number to plot: ").strip())
            metric_name = metric_names.get(metric_number)
            if metric_name:
                save_choice = input("Do you want to save the plot? (yes/no): ").strip().lower()
                if save_choice == "yes" or save_choice == "y":
                    save_path = os.path.join('wandb_data', task)
                    os.makedirs(save_path, exist_ok=True)
                    plot_average_metric_for_task(runs, task, metric_name, save_path=save_path)
                else:
                    plot_average_metric_for_task(runs, task, metric_name)
                    
            else:
                print("Metric number not found.")
        elif choice == "5":
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    main()
