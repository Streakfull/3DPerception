
import os
import random
import numpy as np
import torch
from einops import rearrange
from PIL import Image
from cprint import *
import io
from matplotlib import pyplot as plt


def iou(x_gt, x, thres):
    thres_gt = 0.0

    # compute iou
    # > 0 free space, < 0 occupied
    x_gt_mask = x_gt.clone().detach()
#     x_gt_mask[x_gt < thres_gt] = 0.
#     x_gt_mask[x_gt >= thres_gt] = 1.

    x_mask = x.clone().detach()
    x_mask[x < thres] = 0.
    x_mask[x >= thres] = 1.

    inter = torch.logical_and(x_gt_mask, x_mask)
    union = torch.logical_or(x_gt_mask, x_mask)
    inter = rearrange(inter, 'b d h w -> b (d h w)')
    union = rearrange(union, 'b d h w -> b (d h w)')

    iou = inter.sum(1) / (union.sum(1) + 1e-12)
    return iou


# def iou(gt, pred ,thresh=0.5):
#     pred = pred.clone()
#     gt = gt.clone()
#     # gt[gt<=0.4] = 0
#     # gt[gt>=0.4] = 1
#     pred[pred <= thresh] = 0
#     pred[pred >= thresh] = 1
#     # print((pred!=gt).sum())
#     intersection = torch.sum(pred.mul(gt)).float()
#     union = torch.sum(torch.ge(pred.add(gt), 1)).float()
#     return intersection / union







def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def mkdir(path):
    if not os.path.exists(path):
        cprint.warn(f"- Creating new directory {path}")
        os.makedirs(path)
        return
    cprint.ok(f"- {path} directory found")


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
