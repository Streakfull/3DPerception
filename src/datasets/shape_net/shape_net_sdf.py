import numpy as np
import trimesh
import os
from src.datasets.shape_net.base_shape_net import BaseShapeNet


class ShapeNetSDF(BaseShapeNet):
    def __init__(self, dataset_options, shape_net_options, cat=None):
        super().__init__(
            dataset_options, shape_net_options)

    def get_category_shape_ids(self, category_id):
        ids = os.listdir(self.dataset_path)
        cat_ids = list(
            filter(lambda full_name: full_name.startswith(f"{category_id}_"), ids))
        cat_ids = list(
            map(lambda full_id: full_id.replace(".npz", ""), cat_ids))
        return cat_ids

    def get_shape_info(self, shape_key):
        return shape_key.split("__")

    def __getitem__(self, index):
        shape_key, class_index, class_name, id = super().__getitem__(index)
        sdf_grid = self.get_sdf_grid(shape_key)
        sdf_grid = np.expand_dims(sdf_grid, 0)

        return {
            "sdf": sdf_grid,
            "class_name": class_name,
            "id": id
        }

    def get_sdf_grid(self, shapenet_key):
       # return np.ones((64, 64, 64)) * 0.5
        grid = np.load(
            f"{self.dataset_path}/{shapenet_key}.npz")['arr'].astype(np.float16)

        self.min = np.min(grid)
        self.max = np.max(grid)
        grid = (grid-self.min)/(self.max-self.min)

        return grid

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['sdf'] = batch['sdf'].float().to(device)
