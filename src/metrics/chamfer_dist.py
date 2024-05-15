import numpy as np
from scipy.spatial import KDTree
import open3d as o3d
import mcubes as mc
from src.metrics.base_metric import BaseMetric


class ChamferDistance(BaseMetric):
    def __init__(self, norm=2, apply_center=False, iso=0.02, apply_downsampling=False):
        self.norm = norm
        self.apply_center = apply_center
        self.iso = iso
        self.n_samples = 5000
        self.apply_downsapling = apply_downsampling

    def down_sample(self, pc, samples=5000):
        if (not self.apply_downsapling):
            return pc
        indices = np.arange(pc.shape[0])
        indices = np.random.choice(indices, samples)
        return pc[indices]

    def get_pc_from_sdf(self, sdf):
        vertices, _ = mc.marching_cubes(sdf, self.iso)

        return vertices

    def get_n_samples(self, pcA, pcB):
        min = np.min([self.n_samples, pcA.shape[0], pcB.shape[0]])
        return min

    def calc(self, A, B):
        A = self.get_pc_from_sdf(A)
        B = self.get_pc_from_sdf(B)
        n_samples = self.get_n_samples(A, B)
        A = self.down_sample(A, n_samples)
        B = self.down_sample(B, n_samples)
        return self.calc_dist(A, B)

    def center_pc(self, pc):
        mean = np.mean(pc, axis=0)
        pc = pc-mean
        return pc

    def calc_dist(self, A, B):
        """
        Computes the chamfer distance between two sets of points A and B.
        """
        if (self.apply_center):
            A = self.center_pc(A)
            B = self.center_pc(B)
        tree = KDTree(B)
        dist_A = tree.query(A, p=self.norm)[0]
        tree = KDTree(A)
        dist_B = tree.query(B, p=self.norm)[0]
        if (np.isnan(np.mean(dist_A)) or np.isnan(np.mean(dist_B))):
            return 100
        return np.mean(dist_A) + np.mean(dist_B)


class ChamferDistanceNormal(ChamferDistance):
    def __init__(self, norm=2, apply_center=False, apply_downsampling=False):
        super().__init__(norm, apply_center=apply_center,
                         apply_downsampling=apply_downsampling)

    def calc_normals(self, pc):

        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        o3d_pc.scale(1 / np.max(o3d_pc.get_max_bound() - o3d_pc.get_min_bound()),
                     center=o3d_pc.get_center())
        o3d_pc.estimate_normals()
        # o3d_pc.orient_normals_consistent_tangent_plane(k=30)
        normals = np.asarray(o3d_pc.normals)
        return normals

    def calc_dist(self, A, B):
        """
        Computes the chamfer distance between two sets of points A and B.
        """

        if (self.apply_center):
            A = self.center_pc(A)
            B = self.center_pc(B)

        nA = self.calc_normals(A)
        nB = self.calc_normals(B)

        tree = KDTree(B)
        _, indicesA = tree.query(A, p=self.norm)
        tree = KDTree(A)
        _, indicesB = tree.query(B, p=self.norm)
        normals_dot_product_A = (nB[indicesB] * nA).sum(axis=-1)
        normals_dot_productB = (nA[indicesA] * nB).sum(axis=-1)
        return (np.mean(normals_dot_product_A) + np.mean(normals_dot_productB))/2
