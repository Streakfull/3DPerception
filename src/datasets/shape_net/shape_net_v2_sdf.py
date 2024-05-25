"""PyTorch datasets for loading ShapeNet voxels and ShapeNet SDF from disk"""
import numpy as np
from src.datasets.shape_net.base_shape_net import BaseShapeNet

from src.utils.binvox_rw import read_as_3d_array


class ShapeNetV2SDF(BaseShapeNet):
    def __init__(self, dataset_options, shape_net_options, cat=None):
        super().__init__(
            dataset_options, shape_net_options)

    def __getitem__(self, index):
        shape_key, class_index, class_name, id = super().__getitem__(index)
        sdf,  y = self.get_shape_sdf(shape_key)
        return {
            "sdf": sdf[np.newaxis, :, :, :],
            "label": class_index,
            "class_name": class_name,
            "id": id,

        }

    def get_shape_sdf(self, shapenet_key):
        try:
            sdf = np.load(
                f"{self.dataset_path}/{shapenet_key}/models/model_normalized_size-64_pd-4_tr-6.npy")
            # print(":HIII")
            # sdf = np.clip(sdf, a_min=0, a_max=0.2)
            y = None
        except:
            y = f"{self.dataset_path}/{shapenet_key}"
            sdf = np.zeros((64, 64, 64))

        return sdf, y

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['sdf'] = batch['sdf'].float().to(device)
