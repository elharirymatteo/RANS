import warp as wp
import numpy as np
import torch
import warnings
from omniisaacgymenvs.robots.articulations.utils.Types import DynamicsCfg, ZeroOrderDynamicsCfg, FirstOrderDynamicsCfg, SecondOrderDynamicsCfg, ActuatorCfg

@wp.func
def first_order_dynamics(action: wp.float32, x: wp.float32, dt: wp.float32, T: wp.float32) -> wp.float32:
    """
    This function models first-order dynamics, which gradually adjust the state `x` toward the input `action`.

    Arguments:
        action (wp.float32): Input action or control signal that the system responds to.
        x (wp.float32): Current state or position of the system.
        dt (wp.float32): Time step over which the dynamics are being calculated.
        T (wp.float32): Time constant, determining how fast the system reacts to changes in the input.

    Returns:
        wp.float32: The updated state after applying the first-order dynamics.
    """
    return x + dt * (1. / T) * (action - x)

@wp.func
def second_order_dynamics_p1(x: wp.float32, x_dot: wp.float32, dt: wp.float32) -> wp.float32:
    """
    This function models the first part of second-order dynamics, which updates the position based on the current velocity.

    Arguments:
        x (wp.float32): Current position or state of the system.
        x_dot (wp.float32): Current velocity (time derivative of the position).
        dt (wp.float32): Time step over which the dynamics are being calculated.

    Returns:
        wp.float32: The updated position after applying the first-order dynamics.
    """
    return x + dt * x_dot

@wp.func
def second_order_dynamics_p2(action: wp.float32, x: wp.float32, x_dot: wp.float32, dt: wp.float32, omega_0: wp.float32, zeta: wp.float32) -> wp.float32:
    """
    This function models second-order dynamics, which describes the behavior of a damped harmonic oscillator.

    Arguments:
        action (wp.float32): Input action or force applied to the system.
        x (wp.float32): Current position or state of the system.
        x_dot (wp.float32): Current velocity (time derivative of the position).
        dt (wp.float32): Time step over which the dynamics are being calculated.
        omega_0 (wp.float32): Natural frequency of the system, related to stiffness and mass.
        zeta (wp.float32): Damping ratio, controlling how quickly oscillations decay over time.

    Returns:
        wp.float32: The updated velocity after applying the second-order dynamics.
    """
    return x_dot + dt * (-2. * zeta * omega_0 * x_dot - omega_0*omega_0 * x + omega_0*omega_0 * action)

@wp.func
def scale_action(action: wp.float32, lower_limits: wp.float32, upper_limit: wp.float32) -> wp.float32:
    """
    This function scales an action value from a normalized range (e.g., [-0.5, 0.5]) to a specified range defined by `lower_limits` and `upper_limit`.

    Arguments:
        action (wp.float32): The input action value to be scaled (assumed to be in the range [-0.5, 0.5]).
        lower_limits (wp.float32): The lower limit of the target range.
        upper_limit (wp.float32): The upper limit of the target range.

    Returns:
        wp.float32: The scaled action value in the range [lower_limits, upper_limit].
    """
    return (action + 0.5) * (upper_limit - lower_limits) + lower_limits




@wp.kernel
def apply_first_order_dynamics(actions: wp.array(dtype=wp.float32), xs: wp.array(dtype=wp.float32), dt: wp.float32, T: wp.float32) -> None:
    tid = wp.tid()
    xs[tid] = first_order_dynamics(actions[tid], xs[tid], dt, T)

@wp.kernel
def apply_second_order_dynamics(actions: wp.array2d(dtype=wp.float32), xs: wp.array(dtype=wp.float32), x_dots: wp.array(dtype=wp.float32), dt: wp.float32, omega_0: wp.float32, zeta: wp.float32) -> None:
    tid = wp.tid()
    xs[tid] = second_order_dynamics_p1(xs[tid], x_dots[tid], dt)
    new_x_dot = second_order_dynamics_p2(actions[tid, 0], xs[tid], x_dots[tid], dt, omega_0, zeta)

    if new_x_dot < -300.0:
        new_x_dot = wp.float32(-300.0)
    elif new_x_dot > 300.0:
        new_x_dot = wp.float32(300.0)
        
    x_dots[tid] = new_x_dot

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
        np_z = self._x.to('cpu').numpy()
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
        np_z = self._x_dot.to('cpu').numpy()
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

        return {"x": self._x, "x_dot": self._x_dot}




class Factory:
    def __init__(self) -> None:
        self._creators = {}

    def register(self, dynamics_type: DynamicsCfg, cls: BaseDynamics) -> None:
        self._creators[dynamics_type.__name__] = cls

    def __call__(self, dt:float, num_envs: int, device: str, cfg) -> BaseDynamics:
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
        self._scale_actions = cfg.scale_actions

    def apply_dynamics(self, actions: wp.array2d) -> wp.array:
        if self._scale_actions:
            wp.launch(
                kernel=scale_actions,
                dim=self._num_envs,
                inputs=[
                    actions,
                    self._limits[0],
                    self._limits[1],
                ],
                device=self._device,
            )
        return self._dynamics.apply_dynamics(actions)
    
    def apply_dynamics_torch(self, actions: torch.Tensor) -> torch.Tensor:
        a = wp.from_torch(actions)
        output = self.apply_dynamics(a)
        torch_data = {key: wp.to_torch(value) for key, value in output.items()}
        return torch_data #wp.to_torch(output)

    
    def reset(self) -> None:
        self._dynamics.reset()

    def reset_torch(self, env_ids: torch.Tensor) -> None:
        self._dynamics.reset_torch(env_ids)
    
if __name__ == "__main__":
    # Test the dynamics


    from matplotlib import pyplot as plt
    import numpy as np

    dt = 0.0005
    num_envs = 100
    device = "cuda"
    wp.init()
    cfg = ActuatorCfg(dynamics={"dynamics_type":"second_order", "natural_frequency":100.0, "damping":0.707}, limits={"limits":(-20, 20)})
    actuator = Actuator(dt, num_envs, device, cfg)

    action = wp.zeros((num_envs), device=device, dtype=wp.float32)

    actions = []
    values = []

    for i in range(1000):
        if i < 500:
            action.fill_(0.5)
            output = actuator.apply_dynamics(action)
            values.append(output.to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])
        else:
            action.fill_(-0.5)
            output = actuator.apply_dynamics(action)
            values.append(output.to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])

    t = np.linspace(0, 1000*dt, 1000)
    plt.figure()
    plt.title("Second order dynamics")
    plt.plot(t, actions, label="cmd")
    plt.plot(t, values, label="actuator")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees ($^\circ$)")
    plt.legend()


    dt = 0.005
    cfg = ActuatorCfg(dynamics={"dynamics_type":"first_order", "time_constant":0.2}, limits={"limits":(0, 100)})
    actuator = Actuator(dt, num_envs, device, cfg)

    action = wp.zeros((num_envs), device=device, dtype=wp.float32)

    actions = []
    values = []
    for i in range(1000):
        if i < 500:
            action.fill_(0.5)
            output = actuator.apply_dynamics(action)
            values.append(output.to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])
        else:
            action.fill_(-0.5)
            output = actuator.apply_dynamics(action)
            values.append(output.to('cpu').numpy()[0])
            actions.append(action.to('cpu').numpy()[0])

    t = np.linspace(0, 1000*dt, 1000)
    plt.figure()
    plt.title("First order dynamics")
    plt.plot(t, actions, label="cmd")
    plt.plot(t, values, label="actuator")
    plt.xlabel("Time (s)")
    plt.ylabel("Thrust percentage (%)")
    plt.legend()
    plt.show()


    