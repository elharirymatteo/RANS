import numpy as np
import torch

class BaseWriter:
    def __init__(self, add_noise:bool=False):
        self.add_noise = add_noise
    def _add_noise(self):
        raise NotImplementedError
    def get_data(self, data):
        raise NotImplementedError

class RGBWriter(BaseWriter):
    def get_data(self, data):
        rgb_image = np.frombuffer(data, dtype=np.uint8).reshape(*data.shape, -1)
        rgb_image = np.squeeze(rgb_image)[:, :, :3].transpose((2, 0, 1))
        rgb_image = (rgb_image/255.0).astype(np.float32)
        return torch.from_numpy(rgb_image)

class DepthWriter(BaseWriter):
    def get_data(self, data):
        depth_image = np.frombuffer(data, dtype=np.float32).reshape(*data.shape, -1)
        return torch.from_numpy(depth_image).squeeze().unsqueeze(0)

class SemanticSegmentationWriter(BaseWriter):
    def get_data(self, data):
        raise NotImplementedError
    
class InstanceSegmentationWriter(BaseWriter):
    def get_data(self, data):
        raise NotImplementedError

class ObjectDetectionWriter(BaseWriter):
    def get_data(self, data):
        raise NotImplementedError
        

class WriterFactory:
    """
    Factory class to create tasks."""

    def __init__(self):
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

writer_factory = WriterFactory()
writer_factory.register("RGBWriter", RGBWriter)
writer_factory.register("DepthWriter", DepthWriter)
writer_factory.register("SemanticSegmentationWriter", SemanticSegmentationWriter)
writer_factory.register("InstanceSegmentationWriter", InstanceSegmentationWriter)
writer_factory.register("ObjectDetectionWriter", ObjectDetectionWriter)