import argparse
import os
import subprocess

parser = argparse.ArgumentParser("Processes one or more experiments.")
parser.add_argument(
    "--isaac_path", type=str, default=None, help="Path to the python exec of isaac."
)
args, unknown_args = parser.parse_known_args()

WORKINGDIR = os.getcwd()
s = WORKINGDIR.split("/")[:3]
s = "/".join(s)
if args.isaac_path is None:
    ov_path = os.path.join(s, ".local/share/ov/pkg/isaac_sim-2023.1.1/python.sh")
else:
    ov_path = args.isaac_path

# Define the parameters for the experiments
tasks = ["ASV/GoToPosition", "ASV/GoToPose", "ASV/GoThroughPositionSequence"]  # Add more tasks if needed
# generate 10 seeds
seeds = [i for i in range(10)]

# Define the static parameters
static_params = {
    "headless": "True",
    "train": "ASV/ASV_PPOcontinuous_MLP",
    "num_envs": "4096",
    "max_iterations": "2000",
    "wandb_activate": "True",
    "wandb_entity": "spacer-rl",
    "wandb_project": "test_mt_xpt"
}

# Loop through each combination of task and seed
for task in tasks:
    for seed in seeds:
        experiment_name = f"{task.split('/')[-1]}_{static_params['train'].split('/')[0]}_seed{seed}"
        
        # Construct the command to execute the experiment
        cmd = [ov_path, "scripts/rlgames_train_RANS.py"]
        cmd.append(f"experiment={experiment_name}")
        cmd.append(f"task={task}")
        cmd.append(f"seed={seed}")
        for arg, value in static_params.items():
            cmd.append(f"{arg}={value}")
        
        # Debugging: Print constructed command
        print(f'Constructed command: {" ".join(cmd)}')

        # Execute the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Print command output and errors
        print(f"Output:\n{result.stdout}")
        print(f"Errors:\n{result.stderr}")

        # Check if there are errors
        if result.returncode != 0:
            print(f"Command failed with return code {result.returncode}")
            break
