# Domain Randomization

Unlike the regular version of OmniIsaacGymEnv, this modified version chooses to apply domain randomization 
directly inside the task. This is done so that different parameters can receive different level of noise.
For instance, the state is composed of unnormalized angular values and linear velocity values, both of which 
have largely different scales. Furthermore, the domain randomization we apply here is not limited to noise 
on actions or observations, but we also offer the possibility to randomize the mass of the system,
or apply forces and torques directly onto the system.

All the parameters to add domaine randomization onto the system must be added under the `task.env.disturbances`
flag inside the configuration file. As of today, we support the following disturbances:
 - `force_disturbance` it applies random amount of forces at the system origin.
 - `torque_disturbance` it applies random amount of torque at the system origin.
 - `mass_disturbance` it changes the mass, and center of mass of the system.
 - `observations_disturbance` it adds noise onto the obervations.
 - `actions_disturbance` it adds noise onto the actions.


## Applying disturbances

In the following, we will go over the different parameters available for the disturbances and how to set them.
All the disturbances build ontop of a scheduler, and a sampler.
The scheduler regulates how quickly the disturbances should take effect during the training.
The sampler allows to randomly pick the amount of disturbance that should be apply on each environment.
A detailed explenation of the schedulers and samplers can be found in the curriculum documentation [LINK].

### Force disturbance
This disturbance applies a force on the system. By default, the force is applied at the root/origin of the body.
This behavior can be adjusted by modifying the body on which the force is applied. When setting the parameters
for the disturbance the user will select the magnitude of the force. It will then be randomly applied in a plane,
or on a sphere. Practically this is done by sampling a radius value (that is the magnitude) using the scheduler
and sampler. Then a theta value (for a 2D problem), or a theta and phi value (for a 3D problem), are sampled
uniformly projecting the force accordingly.

Below, is an example of a configuration, please note that all the parameters have default values.
So you do not need to add them unless you want to modify them. In this example, the sampler is
following a `truncated_normal` distribution (a normal distribution with extremas) and the 
scheduler is using a sigmoid growth. We can see that at the begining, there will be almost no force applied,
at the end it is almost uniformly sampled on the [0, 0.5] range.

```yaml
force_disturbance:
  enable: False # Setting this to True will enable this disturbance
  use_sinusoidal_patterns: False # Setting this to True will create none-constant forces.
  min_freq: 0.25
  max_freq: 3
  min_offset: -6
  max_offset: 6
  # Scheduling and sampling of the disturbance
  force_curriculum:
    rate_parameters:
      function: sigmoid
      start: 250
      end: 1250
      extent: 4.5
    sampling_parameters:
      distribution: truncated_normal
      start_mean: 0.0
      start_std: 0.0001
      end_mean: 0.5
      end_std: 0.5
      min_value: 0.0
      max_value: 0.5
```

Setting the `enable` flag to `True` is required for the penalty to be applied. If the flag is left to `False``,
the penalty will not be applied onto the platform.

Setting the `use_sinusoidal_patterns` flag to `False` will mean that each environment will have a constant force applied on it.
If this flag is set to `True`, the force magnitude will be modified depending on the position of the system.
This is meant to recreate attraction and repulsion points. The non-constant force means that recurrent networks will struggle more
to reliably estimate the disturbance.

|![sinusoidal_pattern_0.25](figures/sinusoidal_pattern_025.png) | ![sinusoidal_pattern_3](figures/sinusoidal_pattern_3.png)|
Figure: Two sinusoidal patterns with different frequencies, 0.25: left, 3.0: right.

Please note that the values for the sinusoidal patterns and the magnitude of the force are updated on an environment reset only.
This means that the magnitude of the force will not evolve through an episode.

### Torque disturbance
This disturbance applies a torque on the system. By default, the torque is applied at the root/origin of the body.
This behavior can be adjusted by modifying the body on which the torque is applied. When setting the parameters
for the disturbance the user will select the magnitude of the torque. It will then be randomly applied in a plane,
or on a sphere. Practically this is done by sampling a radius value (that is the magnitude) using the scheduler
and sampler. For a 2D problem, this is the only thing needed, as there is only 1 rotation DoF. For a 3D problem,
theta and phi value (for a 3D problem), are sampledcuniformly projecting the torque accordingly.

Below, is an example of a configuration, please note that all the parameters have default values.
So you do not need to add them unless you want to modify them. In this example, the sampler is
following a `truncated_normal` distribution (a normal distribution with extremas) and the 
scheduler is using a sigmoid growth. We can see that at the begining, there will be almost no torque applied,
at the end it is almost uniformly sampled on the [0, 0.1] range.

```yaml
torque_disturbance:
  enable: False # Setting this to True will enable this disturbance
  use_sinusoidal_patterns: False # Setting this to True will create none-constant forces.
  min_freq: 0.25
  max_freq: 3
  min_offset: -6
  max_offset: 6
  # Scheduling and sampling of the disturbance
  torque_curriculum:
    rate_parameters:
      function: sigmoid
      start: 250
      end: 1250
      extent: 4.5
    sampling_parameters:
      distribution: truncated_normal
      start_mean: 0.0
      start_std: 0.0001
      end_mean: 0.0
      end_std: 0.2
      min_value: -0.1
      max_value: 0.1
```

Setting the `enable` flag to `True` is required for the penalty to be applied. If the flag is left to `False``,
the penalty will not be applied onto the platform.

Setting the `use_sinusoidal_patterns` flag to `False` will mean that each environment will have a constant torque applied on it.
If this flag is set to `True`, the torque magnitude will be modified depending on the position of the system.
This is meant to recreate attraction and repulsion points. The non-constant torque means that recurrent networks will struggle more
to reliably estimate the disturbance.

|![sinusoidal_pattern_0.25](figures/sinusoidal_pattern_025.png) | ![sinusoidal_pattern_3](figures/sinusoidal_pattern_3.png)|
Figure: Two sinusoidal patterns with different frequencies, 0.25: left, 3.0: right.

Please note that the values for the sinusoidal patterns and the magnitude of the torque are updated on an environment reset only.
This means that the magnitude of the torque will not evolve through an episode.

### Mass disturbance
The mass disturbances allows to randomize the mass and the CoM of a rigid body. While it is not currently
possible to randomize the CoM of a rigid bodies inside omniverse, we solve this issue by adding two prismatic
joints to the system, at the end of which lies a fixed mass. All of the other elements inside of
the system are changed to have almost no mass such that the only meaningful contribution to the total system
mass and CoM comes from this movable body.
To randomize the mass value, a scheduler and sampler are used, the mass is directly sampled from it.
For the CoM, another set of scheduler and sampler are used, from it a radius is sampled which can then
be used to move the CoM in a 2D plane by uniformly sampling a theta value, or using in 3D by uniformly
sampling a theta and phi value.

Below is an example configuration, please note that all the parameters have default values.
So you do not need to add them unless you want to modify them. In this example, we can see
that both the mass and the CoM have indepent samplers and rates.

```yaml
mass_disturbance:
  enable: False # Setting this to True will enable this disturbance
  # Scheduling and sampling of the mass disturbance
  mass_curriculum:
    rate_parameters:
      function: sigmoid
      start: 250
      end: 1250
      extent: 4.5
    sampling_parameters:
      distribution: truncated_normal
      start_mean: 5.32 # Initial mass
      start_std: 0.0001 # Low std ensures the mass will remain constand during warmup
      end_mean: 5.32
      end_std: 3.0
      min_value: 3.32
      max_value: 7.32
  # Scheduling and sampling of the CoM disturbance
  com_curriculum:
    rate_parameters:
      function: sigmoid
      start: 250
      end: 1250
      extent: 4.5
    sampling_parameters:
      distribution: truncated_normal
      start_mean: 0.0 # displacement about the resting position of the CoM joints
      start_std: 0.0001 # Low std ensures the mass will remain constand during warmup
      end_mean: 0.25
      end_std: 0.25
      min_value: 0.0
      max_value: 0.25
```

Setting the `enable` flag to `True` is required for the penalty to be applied. If the flag is left to `False``,
the penalty will not be applied onto the platform.

### Observations disturbance
The observation disturbance adds a given type of noise onto the different constituting elements of the 
observation tensor. The noise can be independently controlled and applied or not on 3 type of variables:
 - positions (meters)
 - velocities (meters/s or radians/s)
 - orientation (radians)
For each of them, a scheduler and sampler can be set up, enabling fine control over how the system is exposed
to observation noise during its training.

below is am example configuration, please note that all the parameters have default values.
So you do not need to set them unless you want to modify them.

```yaml
observations_disturbance:
  enable_position_noise: False # Setting this to True will enable this disturbance
  enable_velocity_noise: False # Setting this to True will enable this disturbance
  enable_orientation_noise: False # Setting this to True will enable this disturbance
  # Scheduling and sampling of the position disturbance
  position_curriculum:
    rate_parameters:
      function: sigmoid
      start: 250
      end: 1250
      extent: 4.5
    sampling_parameters:
      distribution: truncated_normal
      start_mean: 0.0
      start_std: 0.0001
      end_mean: 0.0
      end_std: 0.03
      min_value: -0.015
      max_value: 0.015
  # Scheduling and sampling of the velocity disturbance
  velocity_curriculum:
    rate_parameters:
      function: sigmoid
      start: 250
      end: 1250
      extent: 4.5
    sampling_parameters:
      distribution: truncated_normal
      start_mean: 0.0
      start_std: 0.0001
      end_mean: 0.0
      end_std: 0.03
      min_value: -0.015
      max_value: 0.015
  # Scheduling and sampling of the orientation disturbance
  orientation_curriculum:
    rate_parameters:
      function: sigmoid
      start: 250
      end: 1250
      extent: 4.5
    sampling_parameters:
      distribution: truncated_normal
      start_mean: 0.0
      start_std: 0.0001
      end_mean: 0.0
      end_std: 0.05
      min_value: -0.025
      max_value: 0.025
```

Setting the `enable` flag to `True` is required for the penalty to be applied. If the flag is left to `False``,
the penalty will not be applied onto the platform.

### Actions disturbance


## Adding new disturbances
To add new disturbances, 