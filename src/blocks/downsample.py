import torch.nn as nn


class Downsample(nn.Module):
    """Downsampling block from a 3D grid to a lower resolution grid 

    Args:
        nn (nn.Module): Fixed nn.module
    """

    def __init__(self, in_channels, with_conv):
        """Attention block initialization

        Args:
         in_channels (int): number of input channels
         with_conv (boolean): use convultions or average pooling for downsampling
        """
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv3d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=2,
                                  padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x
