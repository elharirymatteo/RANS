import json
import subprocess
import argparse

parser = argparse.ArgumentParser("Processes one or more experiments.")
parser.add_argument("--exps", type=str, nargs="+", default=None, help="List of path to the experiments' config to be ran.")
args, unknown_args = parser.parse_known_args()

for exp in args.exps:
    # Load the configuration file
    with open(exp, 'r') as f:
        experiments = json.load(f)

    # Loop through each experiment and execute it
    for experiment_name, arguments in experiments.items():

        # Construct the command to execute the experiment
        cmd = ['/home/matteo/.local/share/ov/pkg/isaac_sim-2022.2.1/python.sh', 'scripts/rlgames_train.py']
        for arg, value in arguments.items():
            cmd.extend(['{}'.format(arg)+"="+str(value)])
        print(f'Running command: {" ".join(cmd)}')
        # Execute the command
        subprocess.run(cmd)
