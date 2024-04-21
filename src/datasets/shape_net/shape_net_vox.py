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

    def get_items(self):
        items = []
        if (self.is_all_categories()):
            for category in self.classes:
                shape_ids = self.get_category_shape_ids(category)
                items.extend(shape_ids)
            return items
        category_id = self.category_directory_mapping[self.cat]
        return self.get_category_shape_ids(category_id)

    def __getitem__(self, index):
        shape_key = self.items[index]
        voxels = self.get_shape_voxels(shape_key)
        shape_info = shape_key.split("/")
        class_name = self.class_name_mapping[shape_info[0]]

        return {
            "voxels": voxels[np.newaxis, :, :, :],
            "label": class_name,
            "id": shape_info[1]
        }

    def get_shape_voxels(self, shapenet_key):
        with open(self.dataset_path / shapenet_key / "model.binvox", "rb") as fptr:
            voxels = read_as_3d_array(fptr).astype(np.float32)
        return voxels

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['images'] = batch['images'].float().to(device)
        batch['voxels'] = batch['voxels'].float().to(device)
        batch["raw_image"] = batch["raw_image"].float().to(device)

    @staticmethod
    def move_batch_to_device_float(batch, device):
        batch['images'] = batch['images'].float()
        batch['voxels'] = batch['voxels'].float()
        batch["raw_image"] = batch["raw_image"].float()
