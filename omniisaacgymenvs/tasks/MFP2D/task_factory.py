__author__ = "Antoine Richard, Matteo El Hariry"
__copyright__ = (
    "Copyright 2023-24, Space Robotics Lab, SnT, University of Luxembourg, SpaceR"
)
__license__ = "GPL"
__version__ = "2.1.0"
__maintainer__ = "Antoine Richard"
__email__ = "antoine.richard@uni.lu"
__status__ = "development"

from omniisaacgymenvs.tasks.common_3DoF.task_factory import task_factory
from omniisaacgymenvs.tasks.MFP2D.track_linear_angular_velocity import (
    TrackLinearAngularVelocityTask,
)
from omniisaacgymenvs.tasks.MFP2D.track_linear_velocity_heading import (
    TrackLinearVelocityHeadingTask,
)
from omniisaacgymenvs.tasks.MFP2D.close_proximity_dock import (
    CloseProximityDockTask,
)

# Add the system specific tasks
task_factory.register("TrackLinearAngularVelocity", TrackLinearAngularVelocityTask)
task_factory.register("TrackLinearVelocityHeading", TrackLinearVelocityHeadingTask)
task_factory.register("CloseProximityDock", CloseProximityDockTask)