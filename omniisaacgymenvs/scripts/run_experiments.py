import subprocess
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser("Processes one or more experiments.")
parser.add_argument("--exps", type=str, nargs="+", default=None, help="List of path to the experiments' config to be ran.")
parser.add_argument("--isaac_path", type=str, default=None, help="Path to the python exec of isaac.")
args, unknown_args = parser.parse_known_args()

WORKINGDIR = os.getcwd()
s = WORKINGDIR.split("/")[:3]
s = "/".join(s)
if args.isaac_path is None:
    ov_path = os.path.join(s, '.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh')
else:
    ov_path = args.isaac_path

for exp in args.exps:
    # Load the configuration file
    with open(exp, 'r') as f:
        experiments = json.load(f)

    # Loop through each experiment and execute it
    for experiment_name, arguments in experiments.items():

        # Construct the command to execute the experiment
        cmd = [ov_path, 'scripts/rlgames_train.py']
        for arg, value in arguments.items():
            cmd.extend(['{}'.format(arg)+"="+str(value)])
        print(f'Running command: {" ".join(cmd)}')
        # Execute the command
        subprocess.run(cmd)
