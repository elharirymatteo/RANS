import wandb
import sys
import matplotlib.pyplot as plt
import os
import numpy as np

def login_to_wandb():
    wandb.login()

def get_project(entity, project_name):
    api = wandb.Api()
    try:
        return api.runs(f"{entity}/{project_name}")
    except wandb.errors.CommError:
        print("Error: Project not found. Please check the entity and project name.")
        sys.exit(1)

def list_available_runs(runs):
    sorted_runs = sorted(runs, key=lambda run: run.name)
    run_names = {index + 1: run for index, run in enumerate(sorted_runs)}
    print("\nAvailable runs:")
    for index, run in run_names.items():
        print(f"  {index}. {run.name}")
    return run_names

def list_available_metrics(run):
    metrics = sorted(run.summary._json_dict.keys())
    metric_names = {index + 1: metric for index, metric in enumerate(metrics)}
    print("\nAvailable metrics:")
    for index, metric in metric_names.items():
        print(f"  {index}. {metric}")
    return metric_names

def plot_metric_for_runs(runs, metric_name, save_path, plot_type):
    all_histories = []
    
    for run in runs:
        run_name = run.name
        history = run.history(keys=[metric_name])
        if not history.empty:
            steps = history['_step']
            values = history[metric_name]
            all_histories.append((steps, values))
            if plot_type == "all":
                plt.plot(steps, values, label=run.name)

    if plot_type == "average":
        run_name = run.name + "_average"
        all_steps = sorted(set(steps_item for steps, _ in all_histories for steps_item in steps))
        all_values = np.zeros((len(all_histories), len(all_steps)))

        for i, (steps, values) in enumerate(all_histories):
            for j, step in enumerate(all_steps):
                if step in steps.tolist():
                    all_values[i, j] = values[steps.tolist().index(step)] if step in steps.tolist() else np.nan
                else:
                    all_values[i, j] = np.nan
        
        mean_values = np.nanmean(all_values, axis=0)
        std_values = np.nanstd(all_values, axis=0)

        plt.plot(all_steps, mean_values, label="Mean")
        plt.fill_between(all_steps, mean_values - std_values, mean_values + std_values, alpha=0.3, label="Std Dev")

    plt.xlabel('Step')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} over time for runs "{run_name}"')
    plt.legend()

    if save_path:
        plot_filename = os.path.join(save_path, f"{metric_name.replace('/', '_')}.png")
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved as {plot_filename}")
    else:
        plt.show()

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
    run_names = list_available_runs(runs)

    while True:
        print("\nOptions:")
        print("1. Select a run and list its metrics")
        print("2. Filter runs by keyword and plot a specific metric")
        print("3. Exit")
        choice = input("Choose an option (1, 2 or 3): ").strip()

        if choice == "1":
            run_number = int(input("Enter the run number: ").strip())
            run = run_names.get(run_number)
            if run:
                metric_names = list_available_metrics(run)
                metric_number = int(input("Enter the metric number to plot: ").strip())
                metric_name = metric_names.get(metric_number)
                if metric_name:
                    save_choice = input("Do you want to save the plot? (yes/no): ").strip().lower()
                    if save_choice == "yes":
                        save_path = os.path.join('wandb_data', run.name)
                        plot_metric_for_runs([run], metric_name, save_path, "all")
                    else:
                        plot_metric_for_runs([run], metric_name, None, "all")
                else:
                    print("Metric number not found.")
            else:
                print("Run number not found.")
        elif choice == "2":
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
                        plot_metric_for_runs(filtered_runs, metric_name, save_path, plot_type)
                    else:
                        plot_metric_for_runs(filtered_runs, metric_name, None, plot_type)
                else:
                    print("Metric number not found.")
            else:
                print("No runs found with the specified keyword.")            
        elif choice == "3":
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    main()
