import wandb
import sys
import matplotlib.pyplot as plt
import os

def login_to_wandb():
    wandb.login()

def get_project(entity, project_name):
    api = wandb.Api()
    try:
        return api.runs(f"{entity}/{project_name}")
    except wandb.errors.CommError:
        print("Error: Project not found. Please check the entity and project name.")
        sys.exit(1)

def list_charts_for_run(run):
    files = run.files()
    categories = {}
    for file in files:
        category = file.name.split('.')[0]
        if category not in categories:
            categories[category] = []
        categories[category].append(file.name)
    
    for category, charts in categories.items():
        print(f"\nCategory: {category}")
        for chart in charts:
            print(f"  - {chart}")

def show_chart(run, chart_name):
    files = run.files()
    for file in files:
        if chart_name in file.name:
            file.download(replace=True)
            print(f"Downloaded {file.name} from run {run.id}")
            return
    print("Chart not found in the specified run.")


def filter_runs_by_keyword(runs, keyword):
    filtered_runs = [run for run in runs if keyword in run.name or keyword in str(run.config)]
    return filtered_runs

def download_charts_from_filtered_runs(runs, variable_type):
    for run in runs:
        files = run.files()
        for file in files:
            if variable_type in file.name:
                file.download(replace=True)
                print(f"Downloaded {file.name} from run {run.id}")


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

def plot_metrics(run, metric_name, save_path):
    history = run.history(keys=[metric_name])
    if not history.empty:
        plt.plot(history['_step'], history[metric_name])
        plt.xlabel('Step')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name}')
        plot_filename = os.path.join(save_path, f"{metric_name.replace('/', '_')}.png")
        os.makedirs(save_path, exist_ok=True)  # Ensure directory exists
        plt.savefig(plot_filename)
        plt.close()
        print(f"Plot saved as {plot_filename}")
    else:
        print(f"No data found for metric '{metric_name}' in run '{run.name}'.")

def main():
    login_to_wandb()

    default_entity = "spacer-rl"  # Replace with your default entity
    default_project_name = "iclr_benchmark_10s"  # Replace with your default project name

    use_defaults = input("Do you want to use the default entity and project? (yes/no): ").strip().lower()
    if use_defaults == "yes":
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
        print("2. Exit")
        choice = input("Choose an option (1 or 2): ").strip()

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
                        plot_metrics(run, metric_name, save_path)
                    else:
                        print(f"Plot for metric '{metric_name}' will not be saved.")
                else:
                    print("Metric number not found.")
            else:
                print("Run number not found.")
        elif choice == "2":
            print("Exiting the program.")
            sys.exit(0)
        else:
            print("Invalid choice. Please choose a valid option.")

if __name__ == "__main__":
    main()