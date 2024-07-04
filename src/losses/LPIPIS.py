from torch import nn
from src.models.LPIPSVGG import VGG16
from cprint import *
import torch


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv3d(chn_in, chn_out, kernel_size=3, stride=1,
                             padding=1, bias=False), ]
        self.model = nn.Sequential(*layers)


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x/(norm_factor+eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3, 4], keepdim=keepdim)


class LPIPS(nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()
        self.chns = [64, 128, 256, 512, 512]
        self.vgg = VGG16().eval()
        self.lin0 = NetLinLayer(self.chns[0])
        self.lin1 = NetLinLayer(self.chns[1])
        self.lin2 = NetLinLayer(self.chns[2])
        self.lin3 = NetLinLayer(self.chns[3])
        self.lin4 = NetLinLayer(self.chns[4])
        self.ckpt_path = ckpt_path
        self.load_vgg()

        for param in self.parameters():
            param.requires_grad = False

    def load_vgg(self):
        state_dict = torch.load(self.ckpt_path)
        state_dict_copy = {}
        for key in state_dict.keys():
            if "layer6" in key:
                continue
            if "layer7" in key:
                continue
            if "layer8" in key:
                continue
            state_dict_copy[key] = state_dict[key]

        self.vgg.load_state_dict(state_dict_copy)
        cprint.info(f"VGG loaded from {self.ckpt_path}")

    def forward(self, input, target):
        outs0, outs1 = self.vgg(input), self.vgg(target)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(lins[kk].model(diffs[kk]), keepdim=True)
               for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val
