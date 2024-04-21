"""PyTorch datasets for loading ShapeNet voxels and ShapeNet point clouds from disk"""
import numpy as np
import trimesh
from src.datasets.shape_net.base_shape_net import BaseShapeNet


class ShapeNetPoints(BaseShapeNet):
    def __init__(self, dataset_options, shape_net_options, cat=None):
        super().__init__(
            dataset_options, shape_net_options)

    def __getitem__(self, index):
        shape_key, class_index, class_name, id = super().__getitem__(index)
        points = self.get_point_cloud(shape_key)

        return {
            "points": points,
            "label": class_index,
            "class_name": class_name,
            "id": id
        }

    def get_point_cloud(self, shapenet_key):

        path = self.dataset_path / shapenet_key
        mesh = trimesh.load(path).vertices
        return np.transpose(np.float32(mesh))

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['points'] = batch['points'].float().to(device)
        batch['label'] = batch['label'].to(device)

    @staticmethod
    def move_batch_to_device_float(batch, device):
        batch['points'] = batch['points'].float()
        batch['label'] = batch['label']
