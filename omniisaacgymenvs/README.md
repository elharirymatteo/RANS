# How it works: One task to rule them all

This file aims at giving a coarse overview of how our code works.

## How to start the tasks and tune them:

In the latest revision of the tasks, I created them all such that they share the same observation space:\
The observation space is structured like so:
```python
N: number of thrusters

obs = {'state': 10, 'transforms': Nx5, 'masks': N}
```
The values stored inside the `state` are generated as follows:
```python
# heading: The heading of the platform in the global frame
# linear_velocity_x: The linear velocity along the x axis in the global frame
# linear_velocity_y: The linear velocity along the y axis in the global frame
# angular_velocity_z: The angular velocity along the z axis.
# task flag, and integer between 0 and 4.
#   - 0: GoToXY
#   - 1: GoToPose
#   - 2: TrackXYVelocity
#   - 3: TrackXYOVelocity
#   - 4: TrackXYVelocityMatchHeading
# task_data is some bits of data used to fullfil the task.
#   - GoToXY: [error_x, error_y, 0, 0]
#   - GoToPose: [error_x, error_y, cos(error_heading), sin(error_heading)]
#   - TrackXYVelocity: [error_vx, error_vy, 0 , 0]
#   - TrackXYOVelocity: [error_vx, error_vy, error_omega, 0]
#   - TrackXYVelocityMatchHeading: [error_vx, error_vy, cos(error_heading), sin(error_heading)]

[cos(heading), sin(heading), linear_velocity_x, linear_velocity_y, angular_velocity_z, task_flag, task_data_1, task_data_2, task_data_3, task_data_4]
```

The values stored inside the `transforms` are generated as follows:
```python
[[cos(thruster_angle), sin(thruster_angle), dx, dy, thrust_force], ...]
```
If some transforms are not used, if the thruster is killed for instance, then the values inside this array are all null.
The sequence of transform is also organized such that the transforms that are disabled, or used as padding are sent at the end of the sequence.

The values stored inside the `masks` are 0 and 1 indicating if the thrusters are dead or not. 0 the thruster is enabled. 1 the thruster is disabled.

Hence since all task share the same observation space, they can all use the same meta-task. A task that will handle most of the simulation stuff code.
This task is called `MFP2D_Virtual.py`. 

To start the correct task, it uses the parameters given inside the configuration.
For instance, to instantiate the `GoToXY` task you must add the following to the `env` part of the config:
```
  task_parameters: 
    - name: GoToXY
      x_y_tolerance: 0.01
      kill_after_n_steps_in_tolerance: 50 # 10seconds
      max_spawn_dist: 4.0
      min_spawn_dist: 0.5
      kill_dist: 5.0

  reward_parameters:
    - name: GoToXY
      reward_mode: exponential
      exponential_reward_coeff: 0.25
```

To instantiate the `GoToPose` task you must add the following to the `env` part of the config:
```
  task_parameters:
    - name: GoToPose
      x_y_tolerance: 0.01
      heading_tolerance: 0.025
      kill_after_n_steps_in_tolerance: 50 # 10seconds
      max_spawn_dist: 4.0
      min_spawn_dist: 0.5
      kill_dist: 5.0

  reward_parameters:
    - name: GoToPose
      position_reward_mode: exponential
      heading_reward_mode: exponential
      position_exponential_reward_coeff: 0.25
      heading_exponential_reward_coeff: 0.25
      position_scale: 1.0
      heading_scale: 1.0
```

To instantiate the `TrackXYVelocity` task you must add the following to the `env` part of the config:


## How to edit the tasks:
To modify the tasks, there are 3 main files to edit.
- `tasks/virtual_floating_platform/task_parameters`: A file containing dataclasses used to instantiate the task parameters.
- `tasks/virtual_floating_platform/task_rewards`: A file containing dataclasses used to instantiate the reward of the task. It also provide the function used to compute the reward.
- `tasks/virtual_floating_platform/task_factory`: A file containing a factory used to load the correct task based on the given parameters.

To edit the task, a new task must be registered in the factory, a new reward must be added, and a new set of parameters as well. They must all share the same name.
Then to write a task, you can look up how the `tasks/virtual_floating_platform/MFP2D_go_to_xy.py` task is done. (I should have a base class, but I didn't have time).


# How the platform works, and how to tune it:

The code uses the following parameters to automatically generate the floating platform:
```
  platform:
    randomization:
      random_permutation: False
      random_offset: False
      randomize_thruster_position: False
      random_radius: 0.125
      random_theta: 0.125
      randomize_thrust_force: False
      min_thrust_force: 0.5
      max_thrust_force: 1.0
      kill_thrusters: False
      max_thruster_kill: 1

    core:
      mass: 5.0
      CoM: [0,0,0]
      radius: 0.25
      shape: "sphere"
      refinement: 2

    configuration:
      use_four_configurations: False
      num_anchors: 4
      offset: 0.75839816339
      thrust_force: 1.0
      visualize: True
      save_path: "config.png"
```
As you can see, it is segmented into 3 main components, `randomization`, `core`, and `configuration`.

In `core`, you can set the basic parameters:
 - `radius`: the radius of platform, or the distance at which the thruster will be generated.
 - `mass`: the mass of the platform
 - `CoM`: the center of mass of the platform (do not touch)
 - `shape`: the shape of the platform, either `cylinder` or `sphere`.
 - `refinement`: the number of refinements to be applied onto the geometry_prims. (visual only, 2 is good value)

In `configuration` you can tune the thruster setup.
 - `num_anchors`: sets the numbers of thrusters (2 * num_anchors). The thrusters are placed two by two facing opposite direction, and tangeant to a circle. this circle center is aligned with the center of floating platform and its radius is defined inside `core`.
 - `offset`: the offset between the first set of thrusters and the heading of the platform.
 - `thrust_force`: the amount of force that will be applied when firing the thrusters.
 - `use_four_configuration`: this parameter will override `num_anchors`. It will generate 4 configurations, inside a single simualtion. The configurations generated will have 2, 3, 4, and 5 anchors. For this to work as expected the number of environment must be a multiple of 2. Such as 4096, 2048 or 1024. As each configuration will have the exact same number of platforms generated.
 - `visualize`: allows to save a figure showing how the forces are being applied onto the platform when firing all thrusters.
 - `save_path`: the path to which the image should be saved.

In `randomization` you can tune how the configuration will be perturbed:
 - `random_permutation`: enables the shuffling of the transforms.
 - `random_offset`: enables applying a random offset to the thrusters position arounf the platform.
 - `randomize_thrust_force`: randomizes the force applied to the thrusters
 - `min_thrust_force`: the minimal value by which the thrust can be multiplied
 - `max_thrust_force`: the maximum value by which the thrust can be multiplied
 - `ramdomize_thruster_position`: randomizes the position of the thrusters
 - `random_radius`: percentage of displacement relative to the reference distance between the thruster and the center of the platform.
 - `random_theta`: percentage of displacement relative to the reference theta position of the thruster.
 - `kill_thrusters`: kill random thrusters.
 - `max_thruster_kill`: how many thrusters can be killed at most.


# Neural-networks: 
For the new tasks we have three types of networks:
 - `MLP`: A simple MLP network. It uses the `state` as an input.
 - `MLP_thruster`: A simple MLP network. It uses the `state`, and the `masks` as an input. Unlike the regular transformer, it can observe the state of the thrusters.
 - `Transformer`: A transformer network. It uses the `transforms`, the `state`, and the `masks` as an input.
 - `Metamorph`: A transformer network. It uses the `transforms`, the `state`, and the `masks` as an input.

Examples of each of these networks can be seen here:
 - `cfg/train/MFP2D_PPOmulti_dict_MLP`
 - `cfg/train/MFP2D_PPOmulti_dict_MLP_thruster`
 - `cfg/train/MFP2D_PPOmulti_dict_Transformer`
 - `cfg/train/MFP2D_PPOmulti_dict_Metamorph`
