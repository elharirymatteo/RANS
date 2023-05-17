import json
import subprocess

# Load the configuration file
with open('./exp_conf_virt.json', 'r') as f:
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
