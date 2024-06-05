from src.metrics.base_metric import BaseMetric
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
from src.utils.utils_3d import sdf_to_mesh


class Pytorch3DChamferDistance(BaseMetric):
    def __init__(self, norm=2, apply_center=False, iso=0.02, apply_downsampling=False):
        self.norm = norm
        self.apply_center = apply_center
        self.iso = iso
        self.n_samples = 5000
        self.apply_downsapling = apply_downsampling

    def calc(self, A, B):
        A = sdf_to_mesh(A)
        B = sdf_to_mesh(B)
        A = sample_points_from_meshes(A, self.n_samples)
        B = sample_points_from_meshes(B, self.n_samples)
        loss_chamfer, _ = chamfer_distance(A, B)
        return loss_chamfer

    def calc_batch(self, pred, target):
        return self.calc(pred, target)
