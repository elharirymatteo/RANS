import torch
from omniisaacgymenvs.envs.Physics.Utils import *


class HydrodynamicsObject:
    def __init__(
        self,
        dr_params,
        num_envs,
        device,
        params,
    ):
        # TODO: Move to dataclass implementation
        self.linear_damping_base = params["linear_damping"] #TODO: Check if really needed
        self.quadratic_damping_base = params["quadratic_damping"] #TODO: Check if really needed

        self.linear_damping = params["linear_damping"]
        self.quadratic_damping = params["quadratic_damping"]
        self.linear_damping_forward_speed = params["linear_damping_forward_speed"]
        self.offset_linear_damping = params["offset_linear_damping"]
        self.offset_lin_forward_damping_speed = params["offset_lin_forward_damping_speed"]
        self.offset_nonlin_damping = params["offset_nonlin_damping"]
        self.scaling_damping = params["scaling_damping"]
        self.offset_added_mass = params["offset_added_mass"]
        self.scaling_added_mass = params["scaling_added_mass"]

        self._use_drag_randomization = dr_params["use_drag_randomization"]
        # linear_rand range, calculated as a percentage of the base damping coefficients
        self._linear_rand = torch.tensor(
            [
                dr_params["u_linear_rand"] * self.linear_damping[0],
                dr_params["v_linear_rand"] *
                  self.linear_damping[1],
                dr_params["w_linear_rand"] * self.linear_damping[2],
                dr_params["p_linear_rand"] * self.linear_damping[3],
                dr_params["q_linear_rand"] * self.linear_damping[4],
                dr_params["r_linear_rand"] * self.linear_damping[5],
            ],
            device=device,
        )
        self._quad_rand = torch.tensor(
            [
                dr_params["u_quad_rand"] * self.quadratic_damping[0],
                dr_params["v_quad_rand"] * self.quadratic_damping[1],
                dr_params["w_quad_rand"] * self.quadratic_damping[2],
                dr_params["p_quad_rand"] * self.quadratic_damping[3],
                dr_params["q_quad_rand"] * self.quadratic_damping[4],
                dr_params["r_quad_rand"] * self.quadratic_damping[5],
            ],
            device=device,
        )

        self._num_envs = num_envs
        self.device = device
        self.drag = torch.zeros(
            (self._num_envs, 6), dtype=torch.float32, device=self.device
        )

        # damping parameters (individual set for each environment)
        self.linear_damping = torch.tensor(
            [self.linear_damping] * num_envs, device=self.device
        )  # num_envs * 6
        self.quadratic_damping = torch.tensor(
            [self.quadratic_damping] * num_envs, device=self.device
        )  # num_envs * 6
        self.linear_damping_forward_speed = torch.tensor(
            self.linear_damping_forward_speed, device=self.device)
        # damping parameters randomization
        if self._use_drag_randomization:
            # Applying uniform noise as an example
            self.linear_damping += (
                torch.rand_like(self.linear_damping) * 2 - 1
            ) * self._linear_rand
            self.quadratic_damping += (
                torch.rand_like(self.quadratic_damping) * 2 - 1
            ) * self._quad_rand
        # Debug : print the initialized coefficients
        # print("linear_damping: ", self.linear_damping)

        # coriolis
        self._Ca = torch.zeros([6, 6], device=self.device)
        self.added_mass = torch.zeros([num_envs, 6], device=self.device)

        # acceleration
        self._filtered_acc = torch.zeros([6], device=self.device)
        self._last_vel_rel = torch.zeros([6], device=self.device)

        return

    def reset_coefficients(self, env_ids: torch.Tensor, num_resets: int) -> None:
        """
        Resets the drag coefficients for the specified environments.
        Args:
            env_ids (torch.Tensor): Indices of the environments to reset.
        """
        if self._use_drag_randomization:
            # Generate random noise
            noise_linear = (
                torch.rand((len(env_ids), 6), device=self.device) * 2 - 1
            ) * self._linear_rand
            noise_quad = (
                torch.rand((len(env_ids), 6), device=self.device) * 2 - 1
            ) * self._quad_rand

            # Apply noise to the linear and quadratic damping coefficients
            # Use indexing to update only the specified environments
            self.linear_damping[env_ids] = (
                torch.tensor([self.linear_damping_base], device=self.device).expand_as(
                    noise_linear
                )
                + noise_linear
            )
            self.quadratic_damping[env_ids] = (
                torch.tensor(
                    [self.quadratic_damping_base], device=self.device
                ).expand_as(noise_quad)
                + noise_quad
            )
        # Debug : print the updated coefficients
        # print("Updated linear damping for reset envs:", self.linear_damping[env_ids])
        # print(
        #    "Updated quadratic damping for reset envs:", self.quadratic_damping[env_ids]
        # )
        return

    def ComputeDampingMatrix(self, vel):
        """
        // From Antonelli 2014: the viscosity of the fluid causes
        // the presence of dissipative drag and lift forces on the
        // body. A common simplification is to consider only linear
        // and quadratic damping terms and group these terms in a
        // matrix Drb
        """
        # print("vel: ", vel)
        lin_damp = (
            self.linear_damping
            + self.offset_linear_damping
            - (
                self.linear_damping_forward_speed
                + self.offset_lin_forward_damping_speed
            )
        )
        # print("lin_damp: ", lin_damp)
        quad_damp = (
            (self.quadratic_damping + self.offset_nonlin_damping).mT * torch.abs(vel.mT)
        ).mT
        # print("quad_damp: ", quad_damp)
        # scaling and adding both matrices
        damping_matrix = (lin_damp + quad_damp) * self.scaling_damping
        # print("damping_matrix: ", damping_matrix)
        return damping_matrix

    def ComputeHydrodynamicsEffects(
        self, time, quaternions, world_vel, use_water_current, flow_vel
    ):
        rot_mat = quaternion_to_matrix(quaternions)
        rot_mat_inv = rot_mat.mT

        self.local_lin_velocities = getLocalLinearVelocities(
            world_vel[:, :3], rot_mat_inv
        )
        self.local_ang_velocities = getLocalAngularVelocities(
            world_vel[:, 3:], rot_mat_inv
        )

        self.local_velocities = torch.hstack(
            [self.local_lin_velocities, self.local_ang_velocities]
        )

        if use_water_current:
            flow_vel = torch.tensor(flow_vel, device=self.device)

            if flow_vel.dim() == 1:
                flow_vel = flow_vel.unsqueeze(0).expand_as(world_vel[:, :3])

            self.local_flow_vel = getLocalLinearVelocities(flow_vel, rot_mat_inv)
            self.relative_lin_velocities = (
                getLocalLinearVelocities(world_vel[:, :3], rot_mat_inv)
                - self.local_flow_vel
            )
            self.local_velocities = torch.hstack(
                [self.relative_lin_velocities, self.local_ang_velocities]
            )

        # Update damping matrix
        damping_matrix = self.ComputeDampingMatrix(self.local_velocities)

        # Damping forces and torques
        self.drag = -1 * damping_matrix * self.local_velocities

        return self.drag
