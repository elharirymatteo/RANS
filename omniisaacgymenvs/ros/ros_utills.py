import numpy as np
import os

from typing import List, Tuple
#from geometry_msgs.msg import Pose
#import rospy

def enable_ros_extension(env_var:str = "ROS_DISTRO"):
    """
    Enable the ROS extension.
    
    Args:
        env_var (str): The environment variable that contains the ROS distro."""

    import omni.ext

    ROS_DISTRO: str = os.environ.get(env_var, "noetic")
    assert ROS_DISTRO in [
        "noetic",
        "foxy",
        "humble",
    ], f"${env_var} must be one of [noetic, foxy, humble]"

    # Get the extension manager and list of available extensions
    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extensions = extension_manager.get_extensions()

    # Determine the selected ROS extension id
    if ROS_DISTRO == "noetic":
        ros_extension = [ext for ext in extensions if "ros_bridge" in ext["id"]][0]
    elif ROS_DISTRO in "humble":
        ros_extension = [
            ext
            for ext in extensions
            if "ros2_bridge" in ext["id"] and "humble" in ext["id"]
        ][0]
    elif ROS_DISTRO == "foxy":
        ros_extension = [ext for ext in extensions if "ros2_bridge" in ext["id"]][0]

    # Load the ROS extension if it is not already loaded
    if not extension_manager.is_extension_enabled(ros_extension["id"]):
        extension_manager.set_extension_enabled_immediate(ros_extension["id"], True)
    
def angular_velocities(q:np.ndarray, dt:np.ndarray, N:int=1) -> np.ndarray:
    """
    Calculate the angular velocities from the quaternions.
    
    Args:
        q (np.ndarray): The quaternions.
        dt (np.ndarray): The time difference between each quaternion.
        
    Returns:
        np.ndarray: The angular velocities."""
    
    q = q[0::N]
    return (2 / dt) * np.array([
        q[:-1,0]*q[1:,1] - q[:-1,1]*q[1:,0] - q[:-1,2]*q[1:,3] + q[:-1,3]*q[1:,2],
        q[:-1,0]*q[1:,2] + q[:-1,1]*q[1:,3] - q[:-1,2]*q[1:,0] - q[:-1,3]*q[1:,1],
        q[:-1,0]*q[1:,3] - q[:-1,1]*q[1:,2] + q[:-1,2]*q[1:,1] - q[:-1,3]*q[1:,0]])


def derive_velocities(time_buffer:list, pose_buffer: list) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive the velocities from the pose and time buffers.
    
    Args:
        time_buffer (List[rospy.Time]): The time buffer.
        pose_buffer (List[Pose]): The pose buffer.
        
    Returns:
        Tuple(np.ndarray, np.ndarray): The linear and angular velocities."""
    
    dt = (time_buffer[-1] - time_buffer[0]).to_sec() # Time difference between first and last pose
    # Calculate linear velocities
    linear_positions = np.array([[pose.pose.position.x, pose.pose.position.y, pose.pose.position.z] for pose in pose_buffer])
    linear_velocities = np.diff(linear_positions, axis=0) / (dt/len(time_buffer))
    average_linear_velocity = np.mean(linear_velocities, axis=0)

    # Calculate angular velocities
    angular_orientations = np.array([[pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z] for pose in pose_buffer])
    dt_buff = np.ones((angular_orientations.shape[0] - 1)) * dt / (angular_orientations.shape[0] - 1)
    ang_vel = angular_velocities(angular_orientations, dt_buff)
    average_angular_velocity = np.mean(ang_vel, axis=1)

    return average_linear_velocity, average_angular_velocity