__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from inspect import isfunction
import dataclasses
import torch
import math

####################################################################################################
# Curriculum growth functions
####################################################################################################


def curriculum_linear_growth(
    step: int = 0, start: int = 0, end: int = 1000, **kwargs
) -> float:
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


def curriculum_sigmoid_growth(
    step: int = 0, start: int = 100, end: int = 1000, extent: float = 3, **kwargs
) -> float:
    """
    Generates a curriculum with a sigmoid growth rate.

    Args:
        step (int): Current step.
        start (int): Start step.
        end (int): End step.
        extent (float, optional): Extent of the sigmoid function.
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

    rate = (
        math.tanh(((extent * 2 * current / relative_end) - extent) / 2)
        - math.tanh(-extent / 2)
    ) / (math.tanh(extent / 2) - math.tanh(-extent / 2))

    return rate


def curriculum_pow_growth(
    step: int = 0, start: int = 0, end: int = 1000, alpha: float = 2.0, **kwargs
) -> float:
    """
    Generates a curriculum with a power growth rate.

    Args:
        step (int): Current step.
        start (int): Start step.
        end (int): End step.
        alpha (float, optional): Exponent of the power function.
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

    rate = (current / relative_end) ** alpha
    return rate


####################################################################################################
# Curriculum sampling functions
####################################################################################################


def norm_cdf(x: float) -> float:
    """
    Computes standard normal cumulative distribution function
    Args:
        x (float): Input value.
    Returns:
        float: Value of the standard normal cumulative distribution function
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def truncated_normal(
    n: int = 1,
    mean: float = 0.0,
    std: float = 0.5,
    min_value: float = 0.0,
    max_value: float = 1.0,
    device: str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """
    Method based on https://github.com/pytorch/pytorch/blob/a40812de534b42fcf0eb57a5cecbfdc7a70100cf/torch/nn/init.py#L22
    Values are generated by using a truncated uniform distribution and
    then using the inverse CDF for the normal distribution.

    Args:
        n (int, optional): Number of samples to generate.
        mean (float, optional): Mean of the normal distribution.
        std (float, optional): Standard deviation of the normal distribution.
        min_value (float, optional): Minimum value of the truncated distribution.
        max_value (float, optional): Maximum value of the truncated distribution.
        device (str, optional): Device to use for the tensor.
        **kwargs: Additional arguments.

    Returns:
        torch.Tensor: Tensor with values from a truncated normal distribution.
    """

    tensor = torch.zeros((n), dtype=torch.float32, device=device)
    # Get upper and lower cdf values
    l = norm_cdf((min_value - mean) / std)
    u = norm_cdf((max_value - mean) / std)

    # Uniformly fill tensor with values from [l, u], then translate to
    # [2l-1, 2u-1].
    tensor.uniform_(2 * l - 1, 2 * u - 1)

    # Use inverse cdf transform for normal distribution to get truncated
    # standard normal
    tensor.erfinv_()

    # Transform to proper mean, std
    tensor.mul_(std * math.sqrt(2.0))
    tensor.add_(mean)

    # Clamp to ensure it's in the proper range
    tensor.clamp_(min=min_value, max=max_value)
    return tensor


def normal(
    n: int = 1,
    mean: float = 0.0,
    std: float = 0.5,
    device: str = "cpu",
    **kwargs,
) -> torch.Tensor:
    """
    Generates a tensor with values from a normal distribution.

    Args:
        n (int, optional): Number of samples to generate.
        mean (float, optional): Mean of the normal distribution.
        std (float, optional): Standard deviation of the normal distribution.
        device (str, optional): Device to use for the tensor.
        **kwargs: Additional arguments.

    Returns:
        torch.Tensor: Tensor with values from a normal distribution.
    """

    return torch.normal(mean, std, (n), device=device)


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


####################################################################################################
# Function dictionaries
####################################################################################################

RateFunctionDict = {
    "none": lambda step, start, end, **kwargs: 1.0,
    "linear": curriculum_linear_growth,
    "sigmoid": curriculum_sigmoid_growth,
    "pow": curriculum_pow_growth,
}

SampleFunctionDict = {
    "uniform": uniform,
    "normal": normal,
    "truncated_normal": truncated_normal,
}


@dataclasses.dataclass
class CurriculumRateParameters:
    start: int = 50
    end: int = 1000
    function: str = "none"
    extent: float = 3
    alpha: float = 2.0

    def __post_init__(self):
        assert self.start >= 0, "Start must be greater than 0"
        assert self.end > 0, "End must be greater than 0"
        assert self.start < self.end, "Start must be smaller than end"
        assert self.function in [
            "none",
            "linear",
            "sigmoid",
            "pow",
        ], "Function must be linear, sigmoid or pow"
        assert self.extent > 0, "Extent must be greater than 0"
        assert self.alpha > 0, "Alpha must be greater than 0"
        self.function = RateFunctionDict[self.function]
        self.kwargs = {
            key: value for key, value in self.__dict__.items() if not isfunction(value)
        }

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


@dataclasses.dataclass
class CurriculumSamplingParameters:
    distribution: str = "uniform"
    start_min_value: float = 0.0  # uniform only
    start_max_value: float = 0.0  # uniform only
    end_min_value: float = 0.0  # uniform only
    end_max_value: float = 0.0  # uniform only
    start_mean: float = 0.0  # normal and truncated_normal only
    start_std: float = 0.0  # normal and truncated_normal only
    end_mean: float = 0.0  # normal and truncated_normal only
    end_std: float = 0.0  # normal and truncated_normal only
    min_value: float = 0.0  # truncated_normal only
    max_value: float = 0.0  # truncated_normal only

    def __post_init__(self):
        assert (
            self.min_value <= self.max_value
        ), "Min value must be smaller than max value"
        assert (
            self.start_min_value <= self.start_max_value
        ), "Min value must be smaller than max value"
        assert (
            self.end_min_value <= self.end_max_value
        ), "Min value must be smaller than max value"
        assert self.start_std >= 0, "Standard deviation must be greater than 0"
        assert self.end_std >= 0, "Standard deviation must be greater than 0"
        assert self.distribution in [
            "uniform",
            "normal",
            "truncated_normal",
        ], "Distribution must be uniform, normal or truncated_normal"
        self.function = SampleFunctionDict[self.distribution]


@dataclasses.dataclass
class CurriculumParameters:
    rate_parameters: CurriculumRateParameters = dataclasses.field(default_factory=dict)
    sampling_parameters: CurriculumSamplingParameters = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        self.rate_parameters = CurriculumRateParameters(**self.rate_parameters)
        self.sampling_parameters = CurriculumSamplingParameters(
            **self.sampling_parameters
        )


class CurriculumSampler:
    def __init__(
        self,
        curriculum_parameters: CurriculumParameters,
    ):
        self.rp = curriculum_parameters.rate_parameters
        self.sp = curriculum_parameters.sampling_parameters

    def get_rate(self, step: int) -> float:
        """
        Gets the difficulty for the given step.

        Args:
            step (int): Current step.

        Returns:
            float: Current difficulty.
        """

        return self.rp.get(step)

    def get_min(self) -> float:
        """
        Gets the minimum value for the current step.

        Returns:
            float: Minimum value.
        """

        if self.sp.distribution == "truncated_normal":
            return self.sp.start_mean
        elif self.sp.distribution == "normal":
            return self.sp.start_mean
        else:
            return self.sp.start_min_value

    def get_max(self) -> float:
        """
        Gets the maximum value for the current step.

        Returns:
            float: Maximum value.
        """

        if self.sp.distribution == "truncated_normal":
            return self.sp.end_mean
        elif self.sp.distribution == "normal":
            return self.sp.end_mean
        else:
            return self.sp.end_max_value

    def get_min_bound(self) -> float:
        if self.sp.distribution == "truncated_normal":
            return self.sp.min_value
        elif self.sp.distribution == "normal":
            return max(
                [
                    self.sp.end_mean - 2 * self.sp.end_std,
                    self.sp.start_mean - 2 * self.sp.end_std,
                ]
            )
        else:
            return max([self.sp.end_min_value, self.sp.start_min_value])

    def get_max_bound(self) -> float:
        if self.sp.distribution == "truncated_normal":
            return self.sp.max_value
        elif self.sp.distribution == "normal":
            return max(
                [
                    self.sp.end_mean + 2 * self.sp.end_std,
                    self.sp.start_mean + 2 * self.sp.end_std,
                ]
            )
        else:
            return max([self.sp.end_max_value, self.sp.start_max_value])

    def sample(self, n: int, step: int, device: str = "cpu") -> torch.Tensor:
        """
        Samples values from the curriculum distribution.

        Args:
            n (int): Number of samples to generate.
            step (int): Current step.
            device (str): Device to use for the tensor.

        Returns:
            torch.Tensor: Tensor with values from the curriculum distribution.
        """

        # Get the difficulty for the current step
        rate = self.get_rate(step)

        # Sample values from the curriculum distribution
        if self.sp.distribution == "truncated_normal":
            mean = self.sp.start_mean + (self.sp.end_mean - self.sp.start_mean) * rate
            std = self.sp.start_std + (self.sp.end_std - self.sp.start_std) * rate
            return self.sp.function(
                n=n,
                mean=mean,
                std=std,
                min_value=self.sp.min_value,
                max_value=self.sp.max_value,
                device=device,
            )
        elif self.sp.distribution == "normal":
            mean = self.sp.start_mean + (self.sp.end_mean - self.sp.start_mean) * rate
            std = self.sp.start_std + (self.sp.end_std - self.sp.start_std) * rate
            return self.sp.function(n=n, mean=mean, std=std, device=device)
        else:
            min = (
                self.sp.start_min_value
                + (self.sp.end_min_value - self.sp.start_min_value) * rate
            )
            max = (
                self.sp.start_max_value
                + (self.sp.end_max_value - self.sp.start_max_value) * rate
            )
            return self.sp.function(n=n, min_value=min, max_value=max, device=device)
