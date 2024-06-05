import open3d as o3d
import mcubes as mc
import numpy as np
import torch
from src.metrics.base_metric import BaseMetric
from einops import rearrange


class SignedIou(BaseMetric):
    def __init__(self, thresh=0.00):
        self.thresh = thresh

    def calc(self, x, x_gt):
        thres_gt = 0.0

        # compute iou
        # > 0 free space, < 0 occupied
        x_gt_mask = x_gt.clone().detach()
        x_gt_mask[x_gt > thres_gt] = 0.
        x_gt_mask[x_gt <= thres_gt] = 1.

        x_mask = x.clone().detach()
        x_mask[x > self.thresh] = 0.
        x_mask[x <= self.thresh] = 1.

        inter = torch.logical_and(x_gt_mask, x_mask)
        union = torch.logical_or(x_gt_mask, x_mask)
        inter = rearrange(inter, 'b ch d h w -> b (ch d h w)')
        union = rearrange(union, 'b ch d h w -> b (ch d h w)')
        iou = inter.sum(1) / (union.sum(1) + 1e-12)
        return iou.mean()

    def calc_batch(self, pred, target):
        return self.calc(pred, target)
