from omniisaacgymenvs.tasks.MFP.MFP2D_go_to_pose import (
    GoToPoseTask,
)
from omniisaacgymenvs.tasks.MFP.MFP2D_task_rewards import (
    GoToPoseReward,
)
from omniisaacgymenvs.tasks.MFP.MFP2D_task_parameters import (
    GoToPoseParameters,
)

import numpy as np
import unittest
import torch
import math

# =============================================================================
# Default parameters
# =============================================================================


default_params = GoToPoseParameters(
    position_tolerance=0.01,
    heading_tolerance=0.025,
    kill_after_n_steps_in_tolerance=5,
    goal_random_position=0.0,
    max_spawn_dist=6.0,
    min_spawn_dist=3.0,
    kill_dist=8.0,
    spawn_curriculum=False,
    spawn_curriculum_min_dist=0.5,
    spawn_curriculum_max_dist=2.5,
    spawn_curriculum_mode="linear",
    spawn_curriculum_warmup=250,
    spawn_curriculum_end=750,
)

default_rewards = GoToPoseReward(
    position_reward_mode="linear",
    heading_reward_mode="linear",
    position_exponential_reward_coeff=0.25,
    heading_exponential_reward_coeff=0.25,
    position_scale=1.0,
    heading_scale=1.0,
)

default_num_envs = 4
default_device = "cuda:0"

# =============================================================================
# create_stats & update_statistics
# =============================================================================


class TestCreateStats(unittest.TestCase):
    def setUp(self) -> None:

        torch_zeros = lambda: torch.zeros(
            default_num_envs,
            dtype=torch.float,
            device=default_device,
            requires_grad=False,
        )
        self.stats = {
            "position_reward": torch_zeros(),
            "heading_reward": torch_zeros(),
            "position_error": torch_zeros(),
            "heading_error": torch_zeros(),
        }
        self.obj = GoToPoseTask({}, {}, default_num_envs, default_device)
        self.obj._task_parameters = default_params
        self.obj._reward_parameters = default_rewards

        self.position = 1.0
        self.heading = 1.0
        self.position_error = 1.0
        self.heading_error = 1.0
        self.new_stats = {
            "position_reward": torch_zeros() + self.position,
            "heading_reward": torch_zeros() + self.heading,
            "position_error": torch_zeros() + self.position_error,
            "heading_error": torch_zeros() + self.heading_error,
        }

    def test_create_stats(self):
        stats = self.obj.create_stats({})
        self.assertEqual(stats.keys(), self.stats.keys())

    def test_update_statistics(self):
        stats = self.obj.create_stats({})
        self.obj.position_reward = self.stats["position_reward"]
        self.obj.heading_reward = self.stats["heading_reward"]
        self.obj.position_dist = self.stats["position_error"]
        self.obj.heading_dist = self.stats["heading_error"]

        stats = self.obj.update_statistics(self.new_stats)
        self.assertTrue(
            torch.all(stats["position_reward"] == self.new_stats["position_reward"])
        )
        self.assertTrue(
            torch.all(stats["heading_reward"] == self.new_stats["heading_reward"])
        )
        self.assertTrue(
            torch.all(stats["position_error"] == self.new_stats["position_error"])
        )
        self.assertTrue(
            torch.all(stats["heading_error"] == self.new_stats["heading_error"])
        )


# =============================================================================
# get_state_observations
# =============================================================================


class TestGetStateObservation(unittest.TestCase):
    def setUp(self):
        # Current state of the robots
        self.positions = torch.tensor(
            [[0, 0], [1, 1], [2, 2], [-1, -1]], dtype=torch.float, device=default_device
        )
        self.headings = torch.tensor(
            [[0], [np.pi / 2], [np.pi], [-np.pi / 2]],
            dtype=torch.float,
            device=default_device,
        )
        self.orientations = torch.tensor(
            [
                [torch.cos(self.headings[0]), torch.sin(self.headings[0])],
                [torch.cos(self.headings[1]), torch.sin(self.headings[1])],
                [torch.cos(self.headings[2]), torch.sin(self.headings[2])],
                [torch.cos(self.headings[3]), torch.sin(self.headings[3])],
            ],
            dtype=torch.float,
            device=default_device,
        )
        # Targets state of the robots
        self.target_headings = torch.tensor(
            [np.pi * 2, np.pi, np.pi / 2, np.pi / 4],
            dtype=torch.float,
            device=default_device,
        )
        self.target_positions = torch.tensor(
            [[0, 0], [-1, -1], [-2, 2], [-1, -1]],
            dtype=torch.float,
            device=default_device,
        )
        # Expected state observations
        self.expected_position = torch.tensor(
            [[0, 0], [-2, -2], [-4, 0], [0, 0]],
            dtype=torch.float,
            device=default_device,
        )
        self.expected_heading = torch.tensor(
            [0, np.pi / 2, -np.pi / 2, np.pi * 3 / 4],
            dtype=torch.float,
            device=default_device,
        )

        # Recreate the state dict sent to the task
        self.current_state = {
            "position": torch.tensor(
                self.positions, dtype=torch.float, device=default_device
            ),
            "orientation": torch.tensor(
                self.orientations, dtype=torch.float, device=default_device
            ),
            "linear_velocity": torch.zeros(
                (default_num_envs, 2), dtype=torch.float, device=default_device
            ),
            "angular_velocity": torch.zeros(
                (default_num_envs), dtype=torch.float, device=default_device
            ),
        }
        # Generate the task
        self.obj = GoToPoseTask({}, {}, default_num_envs, default_device)
        self.obj._task_parameters = default_params
        self.obj._reward_parameters = default_rewards
        # Overriding the target positions and headings
        self.obj._target_headings = self.target_headings
        self.obj._target_positions = self.target_positions

    def test_get_state_position(self):
        # Generate the state observation to be passed to the agent
        state_observation = self.obj.get_state_observations(self.current_state)
        # Position error in the world frame
        gen_position = state_observation[:, 6:8]

        self.assertTrue(torch.allclose(gen_position, self.expected_position))

    def test_get_state_orientation(self):
        # Generate the state observation to be passed to the agent
        state_observation = self.obj.get_state_observations(self.current_state)
        # Heading error in the world frame (cos(theta), sin(theta))
        gen_heading = torch.arctan2(state_observation[:, 9], state_observation[:, 8])

        self.assertTrue(
            torch.allclose(gen_heading, self.expected_heading, rtol=1e-3, atol=1e-4)
        )


# =============================================================================
# compute_reward & update_kills
# =============================================================================


class TestComputeReward(unittest.TestCase):
    def setUp(self):
        # Current state of the robots
        self.positions = torch.tensor(
            [[0, 0], [1, 1], [2, 2], [-1, -1]], dtype=torch.float, device=default_device
        )
        self.headings = torch.tensor(
            [[0], [np.pi / 2], [np.pi], [-np.pi / 2]],
            dtype=torch.float,
            device=default_device,
        )
        self.orientations = torch.tensor(
            [
                [torch.cos(self.headings[0]), torch.sin(self.headings[0])],
                [torch.cos(self.headings[1]), torch.sin(self.headings[1])],
                [torch.cos(self.headings[2]), torch.sin(self.headings[2])],
                [torch.cos(self.headings[3]), torch.sin(self.headings[3])],
            ],
            dtype=torch.float,
            device=default_device,
        )
        # Targets state of the robots
        self.target_headings = torch.tensor(
            [0, np.pi, np.pi / 2, np.pi / 4],
            dtype=torch.float,
            device=default_device,
        )
        self.target_positions = torch.tensor(
            [[0, 0], [-1, -1], [-2, 2], [-1, -1]],
            dtype=torch.float,
            device=default_device,
        )
        # Expected state observations
        self.expected_position = torch.tensor(
            [[0, 0], [-2, -2], [-4, 0], [0, 0]],
            dtype=torch.float,
            device=default_device,
        )
        self.expected_heading = torch.tensor(
            [0, np.pi / 2, -np.pi / 2, np.pi * 3 / 4],
            dtype=torch.float,
            device=default_device,
        )
        # Recreate the state dict sent to the task
        self.current_state = {
            "position": torch.tensor(
                self.positions, dtype=torch.float, device=default_device
            ),
            "orientation": torch.tensor(
                self.orientations, dtype=torch.float, device=default_device
            ),
            "linear_velocity": torch.zeros(
                (default_num_envs, 2), dtype=torch.float, device=default_device
            ),
            "angular_velocity": torch.zeros(
                (default_num_envs), dtype=torch.float, device=default_device
            ),
        }
        # Generate the task
        self.obj = GoToPoseTask({}, {}, default_num_envs, default_device)
        self.obj._task_parameters = default_params
        self.obj._reward_parameters = default_rewards
        # Overriding the target positions and headings
        self.obj._target_headings = self.target_headings
        self.obj._target_positions = self.target_positions

    def test_get_compute_reward_goal_logic_1(self):
        # Will run 3 steps to check if the condition for goal reached is working
        # Tests shifts in position

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        self.assertTrue(self.obj._goal_reached[0] == 1)
        self.assertTrue(self.obj._goal_reached[1] == 0)
        self.assertTrue(self.obj._goal_reached[2] == 0)
        self.assertTrue(self.obj._goal_reached[3] == 0)

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))

        self.assertTrue(self.obj._goal_reached[0] == 2)
        self.assertTrue(self.obj._goal_reached[1] == 0)
        self.assertTrue(self.obj._goal_reached[2] == 0)
        self.assertTrue(self.obj._goal_reached[3] == 0)

        self.current_state["position"][0, 0] = 2  # moving away from the goal.

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        self.assertTrue(self.obj._goal_reached[0] == 0)
        self.assertTrue(self.obj._goal_reached[1] == 0)
        self.assertTrue(self.obj._goal_reached[2] == 0)
        self.assertTrue(self.obj._goal_reached[3] == 0)

    def test_get_compute_reward_goal_logic_2(self):
        # Will run 3 steps to check if the condition for goal reached is working
        # Tests shifts in heading

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        self.assertTrue(self.obj._goal_reached[0] == 1)
        self.assertTrue(self.obj._goal_reached[1] == 0)
        self.assertTrue(self.obj._goal_reached[2] == 0)
        self.assertTrue(self.obj._goal_reached[3] == 0)

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))

        self.assertTrue(self.obj._goal_reached[0] == 2)
        self.assertTrue(self.obj._goal_reached[1] == 0)
        self.assertTrue(self.obj._goal_reached[2] == 0)
        self.assertTrue(self.obj._goal_reached[3] == 0)

        self.current_state["orientation"][0, 0] = np.cos(
            np.pi / 2
        )  # moving away from the goal.
        self.current_state["orientation"][0, 1] = np.sin(
            np.pi / 2
        )  # moving away from the goal.

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        self.assertTrue(self.obj._goal_reached[0] == 0)
        self.assertTrue(self.obj._goal_reached[1] == 0)
        self.assertTrue(self.obj._goal_reached[2] == 0)
        self.assertTrue(self.obj._goal_reached[3] == 0)

    def test_get_compute_reward_position_dist_is_ok(self):
        # Checks if the position distance is being computed correctly

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))

        expected_dist = torch.sqrt(torch.square(self.expected_position).sum(-1))
        self.assertTrue(
            torch.allclose(self.obj.position_dist, expected_dist, rtol=1e-3, atol=1e-4)
        )

    def test_get_compute_reward_heading_dist_is_ok(self):
        # Checks if the heading distance is being computed correctly

        state_observation = self.obj.get_state_observations(self.current_state)
        # Compute the reward
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))

        expected_dist = torch.abs(self.expected_heading)
        self.assertTrue(
            torch.allclose(self.obj.heading_dist, expected_dist, rtol=1e-3, atol=1e-4)
        )

    def test_update_kills_1(self):
        # Check if the kill condition is being updated correctly
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die1 = self.obj.update_kills()
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die2 = self.obj.update_kills()
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die3 = self.obj.update_kills()
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die4 = self.obj.update_kills()
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die5 = self.obj.update_kills()
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die6 = self.obj.update_kills()

        self.assertTrue(
            torch.all(die1 == torch.tensor([0, 0, 0, 0], device=default_device))
        )
        self.assertTrue(
            torch.all(die2 == torch.tensor([0, 0, 0, 0], device=default_device))
        )
        self.assertTrue(
            torch.all(die3 == torch.tensor([0, 0, 0, 0], device=default_device))
        )
        self.assertTrue(
            torch.all(die4 == torch.tensor([0, 0, 0, 0], device=default_device))
        )
        self.assertTrue(
            torch.all(die5 == torch.tensor([0, 0, 0, 0], device=default_device))
        )
        self.assertTrue(
            torch.all(die6 == torch.tensor([1, 0, 0, 0], device=default_device))
        )

    def test_update_kills_2(self):
        # Check if the kill condition is being updated correctly
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die1 = self.obj.update_kills()
        self.current_state["position"][0, 0] = 20  # moving away from the goal.
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die2 = self.obj.update_kills()
        self.current_state["position"][0, 0] = 0  # moving away from the goal.
        state_observation = self.obj.get_state_observations(self.current_state)
        reward = self.obj.compute_reward(state_observation, torch.zeros(4, 2))
        die3 = self.obj.update_kills()

        self.assertTrue(
            torch.all(die1 == torch.tensor([0, 0, 0, 0], device=default_device))
        )
        self.assertTrue(
            torch.all(die2 == torch.tensor([1, 0, 0, 0], device=default_device))
        )
        self.assertTrue(
            torch.all(die3 == torch.tensor([0, 0, 0, 0], device=default_device))
        )


class TestGetGoals(unittest.TestCase):
    def setUp(self):
        self.num_envs = 1000
        self.obj = GoToPoseTask({}, {}, self.num_envs, default_device)
        self.obj._task_parameters = default_params
        self.obj._target_positions = torch.zeros(
            (self.num_envs, 2), device=default_device
        )
        self.obj._target_headings = torch.zeros(self.num_envs, device=default_device)
        self.target_positions = torch.zeros((self.num_envs, 2), device=default_device)
        self.target_orientations = torch.zeros(
            (self.num_envs, 4), device=default_device
        )

    def test_get_goals(self):
        env_ids = torch.range(
            0, self.num_envs - 1, 1, device=default_device, dtype=torch.int64
        )
        target_positions, target_orientations = self.obj.get_goals(
            env_ids, self.target_positions, self.target_orientations
        )

        # Check if target positions and orientations are updated correctly
        self.assertTrue(torch.all(target_positions[env_ids, :2] != 0))
        self.assertTrue(torch.all(target_orientations[env_ids, 0] != 1))
        self.assertTrue(torch.all(target_orientations[env_ids, 3] != 0))

        # Check if target positions and orientations are within the specified range
        self.assertTrue(
            torch.all(
                torch.abs(target_positions[env_ids, :2])
                <= self.obj._task_parameters.goal_random_position
            )
        )
        self.assertTrue(
            torch.all(
                (torch.abs(target_orientations[env_ids, 0]) <= 1)
                * (torch.abs(target_orientations[env_ids, 3]) <= 1)
            )
        )

        # Check if target headings are within the range of [0, 2*pi]
        self.assertTrue(
            torch.all(
                (self.obj._target_headings[env_ids] >= 0)
                * (self.obj._target_headings[env_ids] <= 2 * math.pi)
            )
        )


if __name__ == "__main__":
    unittest.main()
