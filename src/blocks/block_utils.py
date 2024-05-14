import torch


def Normalize(in_channels):
    """ Pytorch GroupNorm wrapper for normalization. Handles the group norm arguments correctly according to number of channels.

    Args:
        in_channels (int): Number of input channels

    Returns:
      GroupNorm: Correct GroupNorm Class
    """
    if in_channels <= 32:
        num_groups = max(min(1, in_channels // 4), 1)
    else:
        num_groups = 32

    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x):
    """Applies swish activation function https://en.wikipedia.org/wiki/Swish_function

    Args:
        x (tensor D x D): Input Tensor

    Returns:
        tensor (D x D): Output
    """

    # TODO: Use builder and configurable loss function
    return x*torch.sigmoid(x)
