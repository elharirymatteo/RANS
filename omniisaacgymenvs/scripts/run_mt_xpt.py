import argparse
import os
import subprocess
import json

parser = argparse.ArgumentParser("Processes one or more experiments.")
parser.add_argument(
    "--isaac_path", type=str, default=None, help="Path to the python exec of isaac."
)
parser.add_argument(
    "--config_path", type=str, required=True, help="Path to the JSON config file."
)
parser.add_argument(
    "--robot_name", type=str, required=False, help="Specific robot to process."
)
args = parser.parse_args()

# Check robot name validity, but allow for no entry (default train all robots)
valid_robots = ["boat", "floating_platform", "turtle_bot"]
if args.robot_name is not None:
    if args.robot_name not in valid_robots:
        print("Invalid robot name. Valid options are: 'boat', 'floating_platform', 'turtle_bot'")
        exit()

WORKINGDIR = os.getcwd()
s = WORKINGDIR.split("/")[:3]
s = "/".join(s)
if args.isaac_path is None:
    ov_path = os.path.join(s, ".local/share/ov/pkg/isaac_sim-2023.1.1/python.sh")
else:
    ov_path = args.isaac_path

# Load parameters from JSON config file
with open(args.config_path, 'r') as config_file:
    config = json.load(config_file)

# Loop through each robot configuration if a specific robot is not specified, otherwise only process the specified robot
robots_to_process = [args.robot_name] if args.robot_name else config.keys()

for robot_name in robots_to_process:
    if robot_name not in config:
        print(f"Robot {robot_name} not found in the configuration file.")
        continue

    robot_config = config[robot_name]
    print(f"Processing robot: {robot_name}")
    tasks = robot_config["tasks"]
    seeds = robot_config["seeds"]
    static_params = robot_config["static_params"]

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
