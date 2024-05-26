from torch import nn
import torch


class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, codebook_loss, reconstructions, inputs, optimizer_idx=0,
                global_step=0, last_layer=None, cond=None, split="train"):

        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        nll_loss = torch.mean(rec_loss)
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()

        return loss, nll_loss, codebook_loss
