import wandb
import sys

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

def main():
    login_to_wandb()

    default_entity = "spacer-rl"  # Replace with your default entity
    default_project_name = "iclr_benchmark_10s"  # Replace with your default project name

    use_defaults = input("Do you want to use the default entity (" + default_entity + 
                         ") and project (" + default_project_name + ")? (yes/no): ").strip().lower()
    if use_defaults == "yes":
        entity = default_entity
        project_name = default_project_name
    else:
        entity = input("Enter the wandb entity: ").strip()
        project_name = input("Enter the wandb project name: ").strip()

    runs = get_project(entity, project_name)

    # Provide a list of run names to select from
    run_names = {run.name: run.id for run in runs}
    print("\nAvailable runs:")
    for run_name in run_names.keys():
        print(f"  - {run_name}")

    print("\nOptions:")
    print("1. List specific charts available for a specific run")
    print("2. Show one of the previous plots")
    print("3. Use a keyword to filter group of runs and obtain charts for a specific type of variable")
    choice = input("Choose an option (1, 2, or 3): ").strip()

    if choice == "1":
        run_name = input("Enter the run name: ").strip()
        run_id = run_names.get(run_name)
        if run_id:
            run = next((run for run in runs if run.id == run_id), None)
            list_charts_for_run(run)
        else:
            print("Run name not found.")
    elif choice == "2":
        run_name = input("Enter the run name: ").strip()
        run_id = run_names.get(run_name)
        if run_id:
            run = next((run for run in runs if run.id == run_id), None)
            chart_name = input("Enter the chart name to show: ").strip()
            show_chart(run, chart_name)
        else:
            print("Run name not found.")
    elif choice == "3":
        keyword = input("Enter the keyword to filter runs: ").strip()
        variable_type = input("Enter the variable type to obtain charts for (e.g., episode.rewards): ").strip()
        filtered_runs = filter_runs_by_keyword(runs, keyword)
        if filtered_runs:
            download_charts_from_filtered_runs(filtered_runs, variable_type)
        else:
            print("No runs found with the specified keyword.")
    else:
        print("Invalid choice. Please run the script again and choose a valid option.")

if __name__ == "__main__":
    main()
