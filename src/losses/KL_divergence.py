from torch import nn
import torch
import torch.nn.functional as F


class KLDivergence(nn.Module):
    def __init__(self, reduction='mean'):
        super(KLDivergence, self).__init__()
        self.reduction = reduction
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, mu, logvar):
        var = torch.exp(logvar)
        loss = 0.5 * torch.sum(torch.pow(mu, 2)
                               + var - 1.0 - logvar,
                               dim=[1, 2, 3, 4])
        if self.reduction == 'mean':
            return torch.mean(loss, dim=0)
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
