from omniisaacgymenvs.tasks.buoyancy.buoyancy_go_to_xy import GoToXYTask
from omniisaacgymenvs.tasks.buoyancy.buoyancy_go_to_pose import GoToPoseTask
from omniisaacgymenvs.tasks.buoyancy.buoyancy_track_xy_velocity import TrackXYVelocityTask
from omniisaacgymenvs.tasks.buoyancy.buoyancy_track_xyo_velocity import TrackXYOVelocityTask


class TaskFactory:
    """
    Factory class to create tasks."""

    def __init__(self):
        self.creators = {}
    
    def register(self, name: str, task):
        """
        Registers a new task."""
        self.creators[name] = task
        
    def get(self, task_dict: dict, reward_dict: dict, num_envs: int, device: str) -> object:
        """
        Returns a task."""
        assert task_dict["name"] == reward_dict["name"], "The mode of both the task and the reward must match."
        mode = task_dict["name"]
        assert task_dict["name"] in self.creators.keys(), "Unknown task mode."
        return self.creators[mode](task_dict, reward_dict, num_envs, device)


task_factory = TaskFactory()
task_factory.register("GoToXY", GoToXYTask)
task_factory.register("GoToPose", GoToPoseTask)
task_factory.register("TrackXYVelocity", TrackXYVelocityTask)
task_factory.register("TrackXYOVelocity", TrackXYOVelocityTask)
#task_factory.register("TrackXYVelocityHeading", TrackXYVelocityHeadingTask)