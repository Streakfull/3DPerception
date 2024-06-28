from torch import nn
import torch
import torch.nn.functional as F


class KLDivergence(nn.Module):
    def __init__(self, reduction='sum'):
        super(KLDivergence, self).__init__()
        self.reduction = reduction
        # self.kl_loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, mu, logvar):

        # loss = -0.5 * torch.sum(1 + logvar.flatten(1) - mu.flatten(1).pow(2) -
        #                         logvar.flatten(1).exp(), dim=1)
        mu = torch.clamp(mu, -30, 20)
        logvar = torch.clamp(logvar, -30, 20)
        loss = -0.5 * torch.sum(1 + logvar.flatten(1) - mu.flatten(1).pow(2) -
                                logvar.flatten(1).exp(), dim=1)

        # loss = loss / mu.flatten(1).shape[1]
        if self.reduction == 'mean':
            return torch.mean(loss, dim=0)
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))

    # def forward(self, z):
    #     z = z.flatten(1)
    #     target = F.softmax(torch.rand_like(z), dim=1)
    #     z = F.log_softmax(z, dim=1)
    #     return self.kl_loss(z, target)
    # # input =
