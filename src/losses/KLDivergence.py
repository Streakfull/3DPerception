# adapted from https://github.com/fomalhautb/3D-RETR
from torch import nn
import torch


class KLDivergence(nn.Module):
    def __init__(self, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.reduction = reduction

    def forward(self, mu, logvar):
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

        if self.reduction == 'mean':
            return torch.mean(loss, dim=0)
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
