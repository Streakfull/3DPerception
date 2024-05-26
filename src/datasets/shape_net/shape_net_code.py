"""PyTorch datasets for loading ShapeNet voxels and ShapeNet SDF as ori files from disk"""
import numpy as np
from src.datasets.shape_net.base_shape_net import BaseShapeNet
from src.datasets.shape_net.shape_net_v3_sdf import ShapeNetV3SDF
import h5py


class ShapeNetCode(BaseShapeNet):
    def __init__(self, dataset_options, shape_net_options, cat=None):
        super().__init__(
            dataset_options, shape_net_options)

    def __getitem__(self, index):
        shape_key, class_index, class_name, id = super().__getitem__(index)
        sdf = self.get_shape_sdf(shape_key)
        code, code_ix = self.get_codes(shape_key)
        return {
            "sdf": sdf[np.newaxis, :, :, :],
            "label": class_index,
            "class_name": class_name,
            "id": id,
            "code": code,
            "code_ix": code_ix
        }

    @staticmethod
    def move_batch_to_device(batch, device):
        batch['sdf'] = batch['sdf'].float().to(device)
        batch['code'] = batch['code'].float().to(device)
        batch['code_ix'] = batch['code_ix'].to(device)

    def get_codes(self, shape_key):
        code_ix = np.load(f"{self.dataset_path}/{shape_key}/codeix.npy")
        code = np.load(f"{self.dataset_path}/{shape_key}/code.npy")
        return code, code_ix

    def get_shape_sdf(self, shapenet_key):
        sdf = ShapeNetV3SDF.read_sdf(
            f"{self.dataset_path}/{shapenet_key}/ori_sample.h5")
        sdf = np.clip(sdf, a_min=-0.2, a_max=0.2)
        return sdf

    @staticmethod
    def read_sdf(sdf_h5_file):
        h5_f = h5py.File(sdf_h5_file, 'r')
        sdf = h5_f['pc_sdf_sample'][:].astype(np.float32)
        sdf = (sdf).reshape(64, 64, 64)
        # sdf = np.clip(sdf, a_min=-1, a_max=1)
        return sdf
