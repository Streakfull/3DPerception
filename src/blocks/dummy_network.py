import torch.nn.functional as F
import torch.nn as nn
# from empatches import EMPatches
import torch

# A dummy network composed of a deep CNN for classification


class DummyNetwork(nn.Module):
    def __init__(self, configs):
        super().__init__()

    def forward(self, x):
        return x
