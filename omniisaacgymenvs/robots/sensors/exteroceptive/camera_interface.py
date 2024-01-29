import numpy as np
import torch

class BaseCameraInterface:
    """
    Base camera interface class."""
    def __init__(self, add_noise:bool=False):
        """
        Args:
            add_noise (bool): Whether to add noise to the data or not
        """
        self.add_noise = add_noise
    def _add_noise(self):
        """
        Add noise to the data."""
        raise NotImplementedError
    def __call__(self, data):
        """
        Get data from the sensor in torch tensor.
        Args:
            data (Any): data from rep.annotator.get_data()"""
        raise NotImplementedError

class RGBInterface(BaseCameraInterface):
    """
    RGB camera interface class."""
    def __call__(self, data):
        """
        Get rgb data from the sensor in torch tensor.
        Args:
            data (Any): rgb data from rep.annotator.get_data()"""
        rgb_image = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
        rgb_image = np.squeeze(rgb_image)[:, :, :3].transpose((2, 1, 0))
        rgb_image = (rgb_image/255.0).astype(np.float32)
        return torch.from_numpy(rgb_image)

class DepthInterface(BaseCameraInterface):
    """
    Depth camera interface class."""
    def __call__(self, data):
        """
        Get depth data from the sensor in torch tensor.
        Args:
            data (Any): depth data from rep.annotator.get_data()"""
        depth_image = np.frombuffer(data, dtype=np.float32).reshape(*data.shape, -1).transpose((2, 1, 0))
        return torch.from_numpy(depth_image)

class SemanticSegmentationInterface(BaseCameraInterface):
    """
    Semantic segmentation camera interface class."""
    def __call__(self, data):
        """
        Get semantic segmentation data from the sensor in torch tensor.
        Args:
            data (Any): semantic segmentation data from rep.annotator.get_data()"""
        raise NotImplementedError
    
class InstanceSegmentationInterface(BaseCameraInterface):
    """
    Instance segmentation camera interface class."""
    def __call__(self, data):
        """
        Get instance segmentation data from the sensor in torch tensor.
        Args:
            data (Any): instance segmentation data from rep.annotator.get_data()"""
        raise NotImplementedError

class ObjectDetectionInterface(BaseCameraInterface):
    """
    Object detection camera interface class."""
    def __call__(self, data):
        """
        Get object detection data from the sensor in torch tensor.
        Args:
            data (Any): object detection data from rep.annotator.get_data()"""
        raise NotImplementedError
        

class CameraInterfaceFactory:
    """
    Factory class to create tasks."""

    def __init__(self):
        """
        Initialize factor attributes."""
        self.creators = {}

    def register(self, name: str, sensor):
        """
        Registers a new task."""
        self.creators[name] = sensor

    def get(
        self, name: str
    ) -> object:
        """
        Returns a task."""
        assert name in self.creators.keys(), f"{name} not in {self.creators.keys()}"
        return self.creators[name]

camera_interface_factory = CameraInterfaceFactory()
camera_interface_factory.register("RGBInterface", RGBInterface)
camera_interface_factory.register("DepthInterface", DepthInterface)
camera_interface_factory.register("SemanticSegmentationInterface", SemanticSegmentationInterface)
camera_interface_factory.register("InstanceSegmentationInterface", InstanceSegmentationInterface)
camera_interface_factory.register("ObjectDetectionInterface", ObjectDetectionInterface)