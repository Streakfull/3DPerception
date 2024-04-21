"""PyTorch datasets for loading ShapeNet voxels from disk"""
import torch
from pathlib import Path
import json
import numpy as np
import trimesh
import os
from utils.binvox_rw import read_as_3d_array


class ShapeNetVox(torch.utils.data.Dataset):
    num_classes = 13
    class_name_mapping = json.loads(
        Path("datasets/shape_net/shape_info.json").read_text())
    category_directory_mapping = json.loads(
        Path("datasets/shape_net/shape_class_info.json").read_text())
    classes = sorted(class_name_mapping.keys())
    class_names = sorted(class_name_mapping.values())

    def __init__(self, dataset_options, shape_net_options, cat=None):
        self.cat = shape_net_options["category"] if cat is None else cat
        self.is_overfit = dataset_options["is_overfit"]
        self.overfit_size = dataset_options["overfit_size"]
        self.dataset_path = Path(dataset_options["path"])
        self.items = self.get_items()

    def get_items(self):
        items = []
        if (self.is_all_categories()):
            for category in self.classes:
                shape_ids = self.get_category_shape_ids(category)
                items.extend(shape_ids)
            return items
        category_id = ShapeNetVox.category_directory_mapping[self.cat]
        return self.get_category_shape_ids(category_id)

    def get_category_shape_ids(self, category_id):
        ids = os.listdir(
            self.dataset_path / category_id)
        id_categories = map(lambda id: f"{category_id}/{id}", ids)
        return list(id_categories)

    def is_all_categories(self):
        return self.cat == "all"

    def __len__(self):
        if (self.is_overfit and self.overfit_size < len(self.items)):
            return self.overfit_size
        return len(self.items)

    def __getitem__(self, index):
        shape_key = self.items[index]
        voxels = self.get_shape_voxels(shape_key)
        shape_info = shape_key.split("/")
        class_name = ShapeNetVox.class_name_mapping[shape_info[0]]

        return {
            "voxels": voxels[np.newaxis, :, :, :],
            "class": class_name,
            "id": shape_info[1]
        }

    def get_shape_voxels(self, shapenet_key):
        # print(shapenet_key,"KEY")
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
