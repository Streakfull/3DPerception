import open3d as o3d
import mcubes as mc
import numpy as np
import trimesh
from src.metrics.base_metric import BaseMetric


class Iou(BaseMetric):
    def __init__(self, iso=0.02):
        self.iso = iso

    def get_voxels(self, sdf):
        vertices, faces = mc.marching_cubes(sdf, self.iso)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        if (vertices.shape[0] == 0):
            return set()
        res = 1.1875
        voxels = mesh.voxelized(pitch=res)
        voxels = set(tuple(x) for x in voxels.points)
        return voxels

    def calc(self, A, B):
        A = self.get_voxels(A)
        B = self.get_voxels(B)
        iou = len(A.intersection(B)) / \
            len(A.union(B))
        return iou
