"""PyTorch datasets for loading ShapeNet voxels and ShapeNet point clouds from disk"""
import torch
from pathlib import Path
import json
import numpy as np
import trimesh
import os
from utils.binvox_rw import read_as_3d_array
from datasets.shape_net.base_shape_net import BaseShapeNet


class ShapeNetVox(BaseShapeNet):
    def __init__(self, dataset_options, shape_net_options, cat=None):
        super().__init__(
            dataset_options, shape_net_options)

    def __getitem__(self, index):
        shape_key, class_index, class_name, id = super().__getitem__(index)
        voxels = self.get_shape_voxels(shape_key)
        return {
            "voxels": voxels[np.newaxis, :, :, :],
            "label": class_index,
            "class_name": class_name,
            "id": id

        }

    def get_shape_voxels(self, shapenet_key):
        with open(self.dataset_path / shapenet_key / "model.binvox", "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32)
        return voxels

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['voxels'] = batch['voxels'].float().to(device)
        batch['label'] = batch['label'].to(device)

    @staticmethod
    def move_batch_to_device_float(batch, device):
        batch['voxels'] = batch['voxels'].float()
        batch['label'] = batch['label']
