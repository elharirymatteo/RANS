import warp as wp
import numpy as np
import math
import torch
import warnings
from dataclasses import dataclass, field


@dataclass
class DynamicsCfg:
    """
    Base class for dynamics configurations.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
    """

    name: str = "None"


@dataclass
class ZeroOrderDynamicsCfg(DynamicsCfg):
    """
    Zero-order dynamics for the actuators.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
    """

    name: str = "zero_order"


@dataclass
class FirstOrderDynamicsCfg(DynamicsCfg):
    """
    First-order dynamics for the actuators.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
        time_constant (float): The time constant of the dynamics.
    """

    time_constant: float = 0.2
    name: str = "first_order"

    def __post_init__(self) -> None:
        assert self.time_constant > 0, "Invalid time constant, should be greater than 0"


@dataclass
class SecondOrderDynamicsCfg(DynamicsCfg):
    """
    Second-order dynamics for the actuators.

    Args:
        name (str): The name of the dynamics. This is used to identify the
            type of the dynamics. It allows to create the dynamics using
            the factory.
        natural_frequency (float): The natural frequency of the dynamics.
        damping_ratio (float): The damping ratio of the dynamics.
    """

    natural_frequency: float = 100
    damping_ratio: float = 1 / math.sqrt(2)
    name: str = "second_order"

    def __post_init__(self) -> None:
        assert self.natural_frequency > 0, "Invalid natural frequency, should be greater than 0"
        assert self.damping_ratio > 0, "Invalid damping ratio, should be greater than 0"


class TypeFactoryBuilder:
    def __init__(self):
        self.creators = {}

    def register_instance(self, type):
        self.creators[type.__name__] = type

    def register_instance_by_name(self, name, type):
        self.creators[name] = type

    def get_item(self, params):
        assert "name" in list(params.keys()), "The name of the type must be provided."
        assert params["name"] in self.creators, "Unknown type."
        return self.creators[params["name"]](**params)


DynamicsFactory = TypeFactoryBuilder()
DynamicsFactory.register_instance_by_name("zero_order", ZeroOrderDynamicsCfg)
DynamicsFactory.register_instance_by_name("first_order", FirstOrderDynamicsCfg)
DynamicsFactory.register_instance_by_name("second_order", SecondOrderDynamicsCfg)


@dataclass
class ControlLimitsCfg:
    """
    Control limits for the system.

    Args:
        limits (tuple): The limits of the control. The limits should be a tuple
            of length 2 with the first element being the minimum value and the
            second element being the maximum value.
    """

    limits: tuple = field(default_factory=tuple)

    def __post_init__(self) -> None:
        assert self.limits[0] < self.limits[1], "Invalid limits, min should be less than max"
        assert len(self.limits) == 2, "Invalid limits shape, should be a tuple of length 2"


@dataclass
class ActuatorCfg:
    """
    Actuator configuration.

    Args:
        dynamics (dict): The dynamics of the actuator.
        limits (dict): The limits of the actuator.
    """

    dynamics: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.dynamics = DynamicsFactory.get_item(self.dynamics)
        self.limits = ControlLimitsCfg(**self.limits)


@wp.func
def first_order_dynamics(action: wp.float32, x: wp.float32, dt: wp.float32, T: wp.float32) -> wp.float32:
    return x + dt * (1.0 / T) * (action - x)


@wp.func
def second_order_dynamics_p1(x: wp.float32, x_dot: wp.float32, dt: wp.float32) -> wp.float32:
    return x + dt * x_dot


@wp.func
def second_order_dynamics_p2(
    action: wp.float32, x: wp.float32, x_dot: wp.float32, dt: wp.float32, omega_0: wp.float32, zeta: wp.float32
) -> wp.float32:
    return x_dot + dt * (-2.0 * zeta * omega_0 * x_dot - omega_0 * omega_0 * x + omega_0 * omega_0 * action)


@wp.func
def scale_action(action: wp.float32, lower_limits: wp.float32, upper_limit: wp.float32) -> wp.float32:
    return (action + 0.5) * (upper_limit - lower_limits) + lower_limits


@wp.kernel
def apply_first_order_dynamics(
    actions: wp.array(dtype=wp.float32), xs: wp.array(dtype=wp.float32), dt: wp.float32, T: wp.float32
) -> None:
    tid = wp.tid()
    xs[tid] = first_order_dynamics(actions[tid], xs[tid], dt, T)


@wp.kernel
def apply_second_order_dynamics(
    actions: wp.array(dtype=wp.float32),
    xs: wp.array(dtype=wp.float32),
    x_dots: wp.array(dtype=wp.float32),
    dt: wp.float32,
    omega_0: wp.float32,
    zeta: wp.float32,
) -> None:
    tid = wp.tid()
    xs[tid] = second_order_dynamics_p1(xs[tid], x_dots[tid], dt)
    x_dots[tid] = second_order_dynamics_p2(actions[tid], xs[tid], x_dots[tid], dt, omega_0, zeta)


@wp.kernel
def scale_actions(actions: wp.array(dtype=wp.float32), lower_limit: wp.float32, upper_limit: wp.float32):
    tid = wp.tid()
    actions[tid] = scale_action(actions[tid], lower_limit, upper_limit)


class BaseDynamics:
    """
    Base class for the dynamics of the actuators.
    """

    def __init__(self, dt: float, num_envs: int, device: str) -> None:
        """
        Initialize the dynamics.

        Args:
            dt (float): Time step for the simulation.
            num_envs (int): Number of environments.
            device (str): Device to use for the computations.
        """

        self._num_envs = num_envs
        self._device = device

        self._dt = dt
        self._x = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)

    def apply_dynamics(self, commanded_values: wp.array) -> wp.array:
        """
        Apply the dynamics to the commanded values.
        To be implemented by the child classes.

        Args:
            commanded_values (np.array): Commanded values to be processed.

        Returns:
            np.array: Processed values.
        """

        raise NotImplementedError

    def reset(self, env_ids: np.ndarray) -> None:
        """
        Reset the dynamics.
        """
        np_z = self._x.to("cpu").numpy()
        np_z[env_ids] = 0
        self._x = wp.from_numpy(np_z).to(self._device)

    def reset_torch(self, env_ids: torch.Tensor) -> None:
        """
        Reset the dynamics.
        """
        z_t = wp.to_torch(self._x)
        z_t[env_ids] = 0


class ZeroOrderDynamics(BaseDynamics):
    """
    Zero order dynamics, no dynamics applied.
    """

    def __init__(self, dt: float, num_envs: int, device: str, cfg: ZeroOrderDynamicsCfg) -> None:
        super().__init__(dt, num_envs, device)

    def apply_dynamics(self, x: wp.array, **kwargs) -> wp.array:
        """
        Apply the dynamics to the commanded values.

        Args:
            commanded_values (np.array): Commanded values to be processed.

        Returns:
            np.array: Processed values.
        """

        return x


class FirstOrderDynamics(BaseDynamics):
    """
    First order dynamics.
    """

    def __init__(self, dt, num_envs: int, device: str, cfg: FirstOrderDynamicsCfg) -> None:
        """
        Initialize the first order dynamics.

        Args:
            dt (float): Time step for the simulation.
            num_envs (int): Number of environments.
            device (str): Device to use for the computations.
            T (float): Time constant for the dynamics.
        """

        super().__init__(dt, num_envs, device)
        if cfg.time_constant < dt:
            # Warn the user that the time constant is below the dt
            self._T = dt
            warnings.warn("Time constant is below the time step, setting the time constant to the time step.")
        else:
            self._T = cfg.time_constant

    def apply_dynamics(self, actions: wp.array, **kwargs) -> wp.array:
        """
        Apply the dynamics to the commanded values.

        Args:
            commanded_values (np.array): Commanded values to be processed.

        Returns:
            np.array: Processed values.
        """
        wp.launch(
            kernel=apply_first_order_dynamics,
            dim=self._num_envs,
            inputs=[
                actions,
                self._x,
                self._dt,
                self._T,
            ],
            device=self._device,
        )
        return self._x


class SecondOrderDynamics(BaseDynamics):
    """
    Second order dynamics.
    """

    def __init__(self, dt: float, num_envs: int, device: str, cfg: SecondOrderDynamicsCfg) -> None:
        """
        Initialize the second order dynamics.

        Args:
            dt (float): Time step for the simulation.
            num_envs (int): Number of environments.
            device (str): Device to use for the computations.
            omega_0 (float): Natural frequency for the dynamics.
            zeta (float): Damping ratio for the dynamics.
        """
        super().__init__(dt, num_envs, device)
        self._x_dot = wp.zeros((self._num_envs), device=self._device, dtype=wp.float32)
        self._omega_0 = cfg.natural_frequency
        self._zeta = cfg.damping_ratio

    def reset(self, env_ids) -> None:
        """
        Reset the dynamics.
        """
        super().reset(env_ids)
        np_z = self._x_dot.to("cpu").numpy()
        np_z[env_ids] = 0
        self._x_dot = wp.from_numpy(np_z).to(self._device)

    def reset_torch(self, env_ids: torch.Tensor) -> None:
        """
        Reset the dynamics.
        """
        super().reset_torch(env_ids)
        z_t = wp.to_torch(self._x_dot)
        z_t[env_ids] = 0

    def apply_dynamics(self, commanded_values: wp.array) -> wp.array:
        """
        Apply the dynamics to the commanded values.

        Args:
            commanded_values (np.array): Commanded values to be processed.

        Returns:
            np.array: Processed values.
        """
        wp.launch(
            kernel=apply_second_order_dynamics,
            dim=self._num_envs,
            inputs=[
                commanded_values,
                self._x,
                self._x_dot,
                self._dt,
                self._omega_0,
                self._zeta,
            ],
            device=self._device,
        )
        return self._x

    def get_derivate(self):
        return self._x_dot


class Factory:
    def __init__(self) -> None:
        self._creators = {}

    def register(self, dynamics_type: DynamicsCfg, cls: BaseDynamics) -> None:
        self._creators[dynamics_type.__name__] = cls

    def __call__(self, dt: float, num_envs: int, device: str, cfg) -> BaseDynamics:
        if cfg.__class__.__name__ in self._creators:
            return self._creators[cfg.__class__.__name__](dt, num_envs, device, cfg)
        else:
            raise ValueError(f"Invalid dynamics type: {cfg.__class__.__name__}")


dynamics_factory = Factory()
dynamics_factory.register(ZeroOrderDynamicsCfg, ZeroOrderDynamics)
dynamics_factory.register(FirstOrderDynamicsCfg, FirstOrderDynamics)
dynamics_factory.register(SecondOrderDynamicsCfg, SecondOrderDynamics)


class Actuator:
    def __init__(self, dt: float, num_envs: int, device: str, cfg: ActuatorCfg) -> None:
        self._num_envs = num_envs
        self._device = device
        self._dynamics = dynamics_factory(dt, num_envs, device, cfg.dynamics)
        self._limits = cfg.limits.limits

    def apply_dynamics(self, actions: wp.array) -> wp.array:
        breakpoint()
        wp.launch(
            kernel=scale_actions,
            dim=self._num_envs,
            inputs=[
                actions,
                self._limits[0],
                self._limits[1]
            ],
            device=self._device
        )
        return self._dynamics.apply_dynamics(actions)

    def get_derivate(self):
        if isinstance(self._dynamics, SecondOrderDynamics):
            return self._dynamics.get_derivate()
        else:
            return None

    def apply_dynamics_torch(self, actions: torch.Tensor) -> torch.Tensor:
        a = wp.from_torch(actions)
        output = self.apply_dynamics(a)
        return wp.to_torch(output)

    def reset(self) -> None:
        self._dynamics.reset()

    def reset_torch(self, env_ids: torch.Tensor) -> None:
        self._dynamics.reset_torch(env_ids)


if __name__ == "__main__":
    # Test the dynamics

    from matplotlib import pyplot as plt
    import numpy as np

    dt = 0.02
    num_envs = 4
    inertia = 0.001
    device = "cuda"
    wp.init()
    cfg = ActuatorCfg(
        dynamics={"name": "second_order", "natural_frequency": 10.0, "damping_ratio": 1.0},
        limits={"limits": (-100, 100)},
    )
    actuator = Actuator(dt, num_envs, device, cfg)

    action = wp.zeros((num_envs), device=device, dtype=wp.float32)

    actions = []
    values = []
    accels = []

    for i in range(1000):
        if i < 500:
            action.fill_(0.5)
            output = actuator.apply_dynamics(action)
            accel = actuator.get_derivate()
            values.append(output.to("cpu").numpy()[0])
            accels.append(accel.to("cpu").numpy()[0])
            actions.append(action.to("cpu").numpy()[0])
        else:
            action.fill_(-0.5)
            output = actuator.apply_dynamics(action)
            accel = actuator.get_derivate()
            values.append(output.to("cpu").numpy()[0])
            accels.append(accel.to("cpu").numpy()[0])
            actions.append(action.to("cpu").numpy()[0])

    t = np.linspace(0, 1000 * dt, 1000)
    plt.figure()
    plt.title("Second order dynamics")
    plt.plot(t, actions, label="cmd")
    plt.plot(t, values, label="actuator")
    plt.plot(t, accels, label="acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees ($^\circ$)")
    plt.legend()

    # dt = 0.005
    # cfg = ActuatorCfg(dynamics={"name": "first_order", "time_constant": 0.2}, limits={"limits": (0, 100)})
    # actuator = Actuator(dt, num_envs, device, cfg)

    # action = wp.zeros((num_envs), device=device, dtype=wp.float32)

    # actions = []
    # values = []
    # for i in range(1000):
    #     if i < 500:
    #         action.fill_(0.5)
    #         output = actuator.apply_dynamics(action)
    #         values.append(output.to("cpu").numpy()[0])
    #         actions.append(action.to("cpu").numpy()[0])
    #     else:
    #         action.fill_(-0.5)
    #         output = actuator.apply_dynamics(action)
    #         values.append(output.to("cpu").numpy()[0])
    #         actions.append(action.to("cpu").numpy()[0])

    # t = np.linspace(0, 1000 * dt, 1000)
    # plt.figure()
    # plt.title("First order dynamics")
    # plt.plot(t, actions, label="cmd")
    # plt.plot(t, values, label="actuator")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Thrust percentage (%)")
    # plt.legend()
    # plt.show()