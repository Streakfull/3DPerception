import torch
from torch import nn
from src.blocks.block_utils import Normalize, nonlinearity


class ResnetBlock(nn.Module):
    """ResnetBlock implementation with skip connections. 
       x -> [GroupNorm,Conv,Swish] -> [GroupNorm,Dropout,Conv,Swish] -> h + x

    Args:
        nn (nn.Module): Fixed NN module
    """

    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        """Initialization of the Resnetblock

        Args:
            in_channels (int): number of input channels
            dropout (float): probability of an element to be zeroed
            out_channels (int, optional): number of output channels. Defaults to None.
            conv_shortcut (bool, optional): uses a 3x3 convultion for the skip connection instead of 1x1 . Defaults to False.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv3d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.norm2 = Normalize(out_channels)
        # self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv3d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv3d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv3d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.conv1(h)
        h = nonlinearity(h)

        h = self.norm2(h)
       # h = self.dropout(h)
        h = self.conv2(h)
        h = nonlinearity(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h
