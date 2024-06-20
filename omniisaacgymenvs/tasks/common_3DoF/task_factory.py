__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_3DoF.go_to_position import GoToPositionTask
from omniisaacgymenvs.tasks.common_3DoF.go_to_pose import GoToPoseTask
from omniisaacgymenvs.tasks.common_3DoF.track_linear_velocity import (
    TrackLinearVelocityTask,
)
from omniisaacgymenvs.tasks.common_3DoF.go_through_position import GoThroughPositionTask
from omniisaacgymenvs.tasks.common_3DoF.go_through_position_seq import GoThroughPositionSequenceTask
from omniisaacgymenvs.tasks.common_3DoF.go_through_pose import GoThroughPoseTask
from omniisaacgymenvs.tasks.common_3DoF.go_through_pose_seq import (
    GoThroughPoseSequenceTask,
)
from omniisaacgymenvs.tasks.common_3DoF.go_through_gate import GoThroughGateTask
from omniisaacgymenvs.tasks.common_3DoF.go_through_gate_seq import (
    GoThroughGateSequenceTask,
)


class TaskFactory:
    """
    Factory class to create tasks."""

    def __init__(self):
        self.creators = {}

    def register(self, name: str, task):
        """
        Registers a new task."""
        self.creators[name] = task

    def get(
        self, task_dict: dict, reward_dict: dict, num_envs: int, device: str
    ) -> object:
        """
        Returns a task."""
        assert (
            task_dict["name"] == reward_dict["name"]
        ), "The mode of both the task and the reward must match."
        mode = task_dict["name"]
        assert task_dict["name"] in self.creators.keys(), "Unknown task mode."
        return self.creators[mode](task_dict, reward_dict, num_envs, device)


task_factory = TaskFactory()
task_factory.register("GoToPosition", GoToPositionTask)
task_factory.register("GoToPose", GoToPoseTask)
task_factory.register("TrackLinearVelocity", TrackLinearVelocityTask)
task_factory.register("GoThroughPosition", GoThroughPositionTask)
task_factory.register("GoThroughPositionSequence", GoThroughPositionSequenceTask)
task_factory.register("GoThroughPose", GoThroughPoseTask)
task_factory.register("GoThroughPoseSequence", GoThroughPoseSequenceTask)
task_factory.register("GoThroughGate", GoThroughGateTask)
task_factory.register("GoThroughGateSequence", GoThroughGateSequenceTask)
