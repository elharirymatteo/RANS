# Curriculum
To prevent penalties, disturbances, or tasks from being to hard from the beginning,
we use simple fixed curriculum strategies. Here fixed denotes that the rate at which
the task becomes harder is not dynamically adapting to the agent's current capacities.
Instead, it relies on the current step to set the difficulty accordingly.

## Parametrizing the curriculum
In the following we present how to setup the different components of our curriculum objects.
A curriculum object is always composed of a scheduler and a sampler:

```yaml
curriculum_parameters:
  rate_parameters:
    [...]
  sampler_parameters:
    [...]
```

The objects come with default parameters which will result in the rate/scheduler always outputing 1.0.

### Setting up the scheduling/rate of the curriculum
To set the schedule or rate, of the curriculum we provide three main functions:
 - a `sigmoid` style growth.
 - a `power` style growth.
 - a `linear` style growth.
 - `none`, the scheduler always returns 1.0.

Below, we provide 4 sample configuration for each of these functions.

```yaml
rate_parameters: # Sigmoid
  function: sigmoid
  start: 0
  end: 1000
  extent: 4.5 # Must be larger than 0.
```

```yaml
rate_parameters: # Power
  function: power
  start: 0
  end: 1000
  alpha: 2.0 # Can be smaller than 1! Must be larger than 0.
```

```yaml
rate_parameters: # Linear
  function: linear
  start: 0
  end: 1000
```

```yaml
rate_parameters: # None
  function: none
```

How the different parameters impact the scheduling of the curiculum is given in the figure below.
Note than once the scheduler reaches 1.0 it means that the highest difficulty has been reached.
The value outputed by the scheduler is always comprised between \[0,1\].
![curriculum_schedulers](figures/curriculum_schedulers.png)
We can see that for the `sigmoid`, large extent, for instance 12, generate sigmoid with a steeper slope,
while smaller extent get closer to the `linear`. Similarly, creating a `power` function with parameter `alpha`
set to 1.0 will generate the exact same curve as the `linear` function. When `alpha` larger than 1.0 will
have a small slope at the beginning and a high slope at the end, and `alpha` smaller than 1.0 will have
the opposite.

### Setting up the sampler of the curriculum
As of now, we provide 3 basic distribution to sample from:
 - `uniform`, a uniform distribution between a max and a min.
 - `normal`, a normal distribution around a mean with a given sigma.
 - `truncated_normal`, a normal distribution with hard boundaries.

Below, we provide 3 sample configurations:

```yaml
sampling_parameters: # Uniform
  distribution: uniform
  start_min_value: -0.1
  start_max_value: 0.1
  end_min_value: -0.3
  end_max_value: 0.3
```

```yaml
sampling_parameters: # Normal
  distribution: normal
  start_mean: 0.0
  start_std: 0.0001
  end_mean: 0.0
  end_std: 0.2
```

```yaml
sampling_parameters: # Truncated normal
  distribution: truncated_normal
  start_mean: 0.0
  start_std: 0.0001
  end_mean: 0.0
  end_std: 0.2
  min_value: -0.1
  max_value: 0.1
```

In the above example, we can see that there is always a start, and an end parameter, be it for the mean, std,
or max and min value of the uniform distribution. Start denotes the distribution as it will be when the 
scheduler/rate output is 0. End denotes the distribution as it will be when the scheduler/rate output is 1.
In between, the distribution will transition from one to the other following the function given to the scheduler.

## Modifying the curriculum
In the following we explain how to add new samplers and schedulers to the current set 
of curriculum. In the futur we plan on expanding the curriculum to support non-fixed steps.

### Modifying the scheduler
Adding a new scheduler is relatively easy and straight forward.
Create a new function inside `tasks.virtual_floating_platform.curriculum_helpers.py`.
Make sure this function has the following header:
```python
def your_new_function(step: int = 0, start: int = 0, end: int = 1000, **kwargs) -> float
```
Note that in practice step can be a float.

Below is our linear function:
```python
def curriculum_linear_growth(step: int = 0, start: int = 0, end: int = 1000, **kwargs) -> float:
    """
    Generates a curriculum with a linear growth rate.

    Args:
        step (int): Current step.
        start (int): Start step.
        end (int): End step.
        **kwargs: Additional arguments.

    Returns:
        float: Rate of growth.
    """

    if step < start:
        return 0.0

    if step > end:
        return 1.0

    current = step - start
    relative_end = end - start

    rate = current / (relative_end)

    return rate
```

Then add this function to the RateFunctionDict, here is an example.
```python
RateFunctionDict = {
    "none": lambda step, start, end, **kwargs: 1.0,
    "linear": curriculum_linear_growth,
    "sigmoid": curriculum_sigmoid_growth,
    "pow": curriculum_pow_growth,
}
```

Finally to call your own function, use the key you set inside the dictionary as 
the `function` parameter in the rate/scheduler config.

But what if you wanted to add more parameters? In theory, there is an automatic parameter collector.
That means that as long as you create functions with named variables, and that these named variables 
match the name of the parameters given to the dataclass, everything should be seemless. With the
notable exception of functions. Below is the automatic parameter collector:
```python
self.kwargs = {
    key: value for key, value in self.__dict__.items() if not isfunction(value)
}
```
This is then process inside the following:
```python
def get(self, step: int) -> float:
    """
    Gets the difficulty for the given step.

    Args:
        step (int): Current step.

    Returns:
        float: Current difficulty.
    """

    return self.function(
        step=step,
        **self.kwargs,
    )
```

### Modifying the sampler
Similarly, a new sampler can be added in the same fashion.
Create a new function inside `tasks.virtual_floating_platform.curriculum_helpers.py`.
This function must follow the following header style:
```python
def your_new_function(n: int = 1, device: str = "cpu", **kwargs) -> torch.Tensor:
```
You can add arguments as you see fit.

Below is our implementation of the uniform sampling:
```python
def uniform(
    n: int = 1,
    min_value: float = 0.0,
    max_value: float = 1.1,
    device: str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """
    Generates a tensor with values from a uniform distribution.

    Args:
        n (int, optional): Number of samples to generate.
        min_value (float, optional): Minimum value of the uniform distribution.
        max_value (float, optional): Maximum value of the uniform distribution.
        device (str, optional): Device to use for the tensor.
        **kwargs: Additional arguments.

    Returns:
        torch.Tensor: Tensor with values from a uniform distribution.
    """

    return torch.rand((n), device=device) * (max_value - min_value) + min_value
```

Proceed to add this function inside the `SampleFunctionDict`:
```python
SampleFunctionDict = {
    "uniform": uniform,
    "normal": normal,
    "truncated_normal": truncated_normal,
}
```

With this done, all that's left to is to define the routine to update the different parameters
given the rate. While this operation could be automated this would likely lead to the overall
code being less flexible. Thus, we require to update the `CurriculumSampler` class.

Inside the `sample` function, you will need to add an if statement that matches your distribution's
name. An example is given below:
```python
elif self.sp.distribution == "normal":
    mean = self.sp.start_mean + (self.sp.end_mean - self.sp.start_mean) * rate
    std = self.sp.start_std + (self.sp.end_std - self.sp.start_std) * rate
    return self.sp.function(n=n, mean=mean, std=std, device=device)
```