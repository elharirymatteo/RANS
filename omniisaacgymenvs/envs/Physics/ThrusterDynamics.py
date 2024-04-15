import torch


class Dynamics:
    def __init__(self, num_envs, device):
        self.num_envs = num_envs
        self.device = device
        self.thrusters = torch.zeros(
            (self.num_envs, 6), dtype=torch.float32, device=self.device
        )
        self.current_forces = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.Reset()

    def update(self, cmd, dt):
        raise NotImplementedError()

    def Reset(self):
        self.current_forces[:, :] = 0.0
        # print(f"current_forces: {self.current_forces}")


class DynamicsZeroOrder(Dynamics):
    def __init__(self, num_envs, device):
        super().__init__(num_envs, device)
        return

    def update(self, cmd):
        return cmd


class DynamicsFirstOrder(Dynamics):
    def __init__(
        self,
        task_cfg,
        num_envs,
        device,
        timeConstant,
        dt,
        numberOfPointsForInterpolation,
        interpolationPointsFromRealDataLeft,
        interpolationPointsFromRealDataRight,
        coeff_neg_commands,
        coeff_pos_commands,
        cmd_lower_range,
        cmd_upper_range,
    ):
        super().__init__(num_envs, device)
        self.tau = timeConstant
        self.idx_matrix = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.dt = dt

        # thruster randomization
        self._use_thruster_randomization = task_cfg["use_thruster_randomization"]
        self._thruster_rand = task_cfg["thruster_rand"]
        self._use_separate_randomization = task_cfg["use_separate_randomization"]
        self._left_rand = task_cfg["left_rand"]
        self._right_rand = task_cfg["right_rand"]
        self.thruster_multiplier = torch.ones(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        self.thruster_left_multiplier = torch.ones(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        self.thruster_right_multiplier = torch.ones(
            (self.num_envs, 1), dtype=torch.float32, device=self.device
        )
        if self._use_thruster_randomization:
            if self._use_separate_randomization:
                self.thruster_left_multiplier = torch.rand(
                    (self.num_envs, 1), dtype=torch.float32, device=self.device
                ) * 2 * self._left_rand + (1 - self._left_rand)
                self.thruster_right_multiplier = torch.rand(
                    (self.num_envs, 1), dtype=torch.float32, device=self.device
                ) * 2 * self._right_rand + (1 - self._right_rand)
            else:
                self.thruster_multiplier = torch.rand(
                    (self.num_envs, 1), dtype=torch.float32, device=self.device
                ) * 2 * self._thruster_rand + (1 - self._thruster_rand)
        # interpolate
        self.commands = torch.linspace(
            cmd_lower_range,
            cmd_upper_range,
            steps=len(interpolationPointsFromRealDataLeft),
            device=self.device,
        )
        self.numberOfPointsForInterpolation = numberOfPointsForInterpolation
        self.interpolationPointsFromRealDataLeft = torch.tensor(
            interpolationPointsFromRealDataLeft, device=self.device
        )
        self.interpolationPointsFromRealDataRight = torch.tensor(
            interpolationPointsFromRealDataRight, device=self.device
        )

        # forces
        self.thruster_forces_before_dynamics = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )
        self.thruster_forces_after_randomization = torch.zeros(
            (self.num_envs, 2), dtype=torch.float32, device=self.device
        )

        # lsm
        self.coeff_neg_commands = torch.tensor(coeff_neg_commands, device=self.device)
        self.coeff_pos_commands = torch.tensor(coeff_pos_commands, device=self.device)

        self.interpolate_on_field_data()

    def reset_thruster_randomization(
        self, env_ids: torch.Tensor, num_resets: int
    ) -> None:
        if self._use_thruster_randomization:
            if self._use_separate_randomization:
                self.thruster_left_multiplier[env_ids] = torch.rand(
                    (num_resets, 1), dtype=torch.float32, device=self.device
                ) * 2 * self._left_rand + (1 - self._left_rand)
                self.thruster_right_multiplier[env_ids] = torch.rand(
                    (num_resets, 1), dtype=torch.float32, device=self.device
                ) * 2 * self._right_rand + (1 - self._right_rand)
            else:
                self.thruster_multiplier[env_ids] = torch.rand(
                    (num_resets, 1), dtype=torch.float32, device=self.device
                ) * 2 * self._thruster_rand + (1 - self._thruster_rand)
        return

    def update(self, thruster_forces_before_dynamics, dt):
        """thrusters dynamics"""

        alpha = torch.exp(torch.tensor((-dt / self.tau), device=self.device))
        self.current_forces[:, :] = (
            self.current_forces[:, :] * alpha
            + (1.0 - alpha) * thruster_forces_before_dynamics
        )

        # debugging
        # print(self.current_forces[:,:])

        return self.current_forces

    def compute_thrusters_constant_force(self):
        """for testing purpose"""

        # turn
        self.thrusters[:, 0] = 400
        self.thrusters[:, 3] = -400

        return self.thrusters

    def interpolate_on_field_data(self):
        """interpolates the data furnished by on-field experiment"""

        self.x_linear_interp = torch.linspace(
            min(self.commands), max(self.commands), self.numberOfPointsForInterpolation
        )
        self.y_linear_interp_left = torch.nn.functional.interpolate(
            self.interpolationPointsFromRealDataLeft.unsqueeze(0).unsqueeze(0),
            size=self.numberOfPointsForInterpolation,
            mode="linear",
            align_corners=True,
        )
        self.y_linear_interp_right = torch.nn.functional.interpolate(
            self.interpolationPointsFromRealDataRight.unsqueeze(0).unsqueeze(0),
            size=self.numberOfPointsForInterpolation,
            mode="linear",
            align_corners=True,
        )
        self.y_linear_interp_left = self.y_linear_interp_left.squeeze(0).squeeze(
            0
        )  # back to dim 1
        self.y_linear_interp_right = self.y_linear_interp_right.squeeze(0).squeeze(
            0
        )  # back to dim 1
        self.n_left = self.numberOfPointsForInterpolation
        self.n_right = self.numberOfPointsForInterpolation

    def get_cmd_interpolated(self, cmd_value):
        """get the corresponding force value in the lookup table of interpolated forces"""
        # Debug: print(cmd_value)
        # print(f"cmd_value: {cmd_value}")
        # cmd_value is size (num_envs,2)
        idx_left = torch.round(((cmd_value[:, 0] + 1) / 2 * self.n_left) - 1).to(
            torch.long
        )
        idx_right = torch.round(((cmd_value[:, 1] + 1) / 2 * self.n_right) - 1).to(
            torch.long
        )
        # print(f"idx_left: {idx_left}")
        # Using indices to gather interpolated forces for each thruster
        self.thruster_forces_before_dynamics[:, 0] = self.y_linear_interp_left[idx_left]
        self.thruster_forces_before_dynamics[:, 1] = self.y_linear_interp_right[
            idx_right
        ]

        # Applying thruster randomization
        if self._use_thruster_randomization:
            if self._use_separate_randomization:
                self.thruster_forces_after_randomization[:, 0] = (
                    self.thruster_forces_before_dynamics[:, 0]
                    * self.thruster_left_multiplier.squeeze()
                )
                self.thruster_forces_after_randomization[:, 1] = (
                    self.thruster_forces_before_dynamics[:, 1]
                    * self.thruster_right_multiplier.squeeze()
                )
            else:
                self.thruster_forces_after_randomization = (
                    self.thruster_forces_before_dynamics * self.thruster_multiplier
                )

    def set_target_force(self, commands):
        """this function get commands as entry and provide resulting forces"""

        # size (num_envs,2)
        self.get_cmd_interpolated(commands)  # every action step

    def update_forces(self):
        # size (num_envs,2)
        if self._use_thruster_randomization:
            self.thrusters[:, [0, 3]] = self.update(
                self.thruster_forces_after_randomization, self.dt
            )  # every simulation step that tracks the target  update_thrusters_forces
        else:
            self.thrusters[:, [0, 3]] = self.update(
                self.thruster_forces_before_dynamics, self.dt
            )

        # Debug: print(self.thrusters[:,[0,3]])
        # print(f"thrusters: {self.thrusters[:,[0,3]]}")
        return self.thrusters

    """function below has to be change to fit multi robots training"""

    def command_to_thrusters_force_lsm(
        self, left_thruster_command, right_thruster_command
    ):
        """This function implement the non-linearity of the thrusters according to a command"""

        T_left = 0
        T_right = 0

        n = len(self.coeff_neg_commands) - 1

        if left_thruster_command < 0:
            for i in range(n):
                T_left += (left_thruster_command ** (n - i)) * self.coeff_neg_commands[
                    i
                ]
            T_left += self.coeff_neg_commands[n]
        elif left_thruster_command >= 0:
            for i in range(n):
                T_left += (left_thruster_command ** (n - i)) * self.coeff_pos_commands[
                    i
                ]
            T_left += self.coeff_pos_commands[n]

        if right_thruster_command < 0:
            for i in range(n):
                T_right += (
                    right_thruster_command ** (n - i)
                ) * self.coeff_neg_commands[i]
            T_right += self.coeff_neg_commands[n]
        elif right_thruster_command >= 0:
            for i in range(n):
                T_right += (
                    right_thruster_command ** (n - i)
                ) * self.coeff_pos_commands[i]
            T_right += self.coeff_pos_commands[n]

        self.thrusters[:, 0] = self.update(T_left, 0.01)
        self.thrusters[:, 3] = self.update(T_right, 0.01)

        return self.thrusters
