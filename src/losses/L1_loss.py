from torch import nn
import torch
import torch.nn.functional as F


class L1(nn.Module):
    def __init__(self, reduction='mean'):
        super(L1, self).__init__()
        self.reduction = reduction

    def forward(self, prediction, target):
        loss = torch.abs(
            prediction.contiguous() - target.contiguous())

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'batch_mean':
            loss = loss.sum() / loss.flatten(1).shape[0]
            return loss
        elif self.reduction == 'none':
            return loss

        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
