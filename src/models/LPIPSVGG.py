from src.models.base_model import BaseModel
from torch import nn
from torch import optim
import torch
import json
import os
from pathlib import Path
from collections import namedtuple


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv3d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm3d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i])
              for i in range(len(in_list))]
    layers += [nn.MaxPool3d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_name_mapping = json.loads(
            Path("src/datasets/shape_net/shape_info.json").read_text())
        self.classes = sorted(self.class_name_mapping.keys())
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([1, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block(
            [64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
       # self.layer6 = vgg_fc_layer(4096, 4096)
       # self.layer7 = vgg_fc_layer(4096, 4096)
        # self.configs = configs
        # Final layer
        # self.layer8 = nn.Linear(4096, 13)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.layer1(x)
        h_relu1_2 = h
        h = self.layer2(h)
        h_relu2_2 = h
        h = self.layer3(h)
        h_relu3_3 = h
        h = self.layer4(h)
        h_relu4_3 = h
        h = self.layer5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_3, h_relu4_3, h_relu5_3)
        return out
