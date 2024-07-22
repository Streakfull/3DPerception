from torch import nn
import torch
from src.losses.LPIPIS import LPIPS


class VQLoss(nn.Module):
    def __init__(self, codebook_weight=1.0, vgg_checkpoint=" logs/VGG/trainFull/2024_07_01_17_50_17/checkpoints/epoch-36.ckpt", perceptual_weight=0):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        if (self.perceptual_weight > 0):
            self.LPIPIS = LPIPS(ckpt_path=vgg_checkpoint).eval()

    def forward(self, codebook_loss, reconstructions, inputs, optimizer_idx=0,
                global_step=0, last_layer=None, cond=None, split="train"):

        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        l1_copy = rec_loss.detach().mean()
        if (self.perceptual_weight > 0):
            p_loss = self.LPIPIS(reconstructions, inputs)
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0, dtype=rec_loss.dtype)

        nll_loss = torch.mean(rec_loss)
        loss = nll_loss + self.codebook_weight * codebook_loss.mean()
        return {"loss": loss, "l1": l1_copy, "codebook": codebook_loss, "p": torch.abs(p_loss.detach().mean())*10}
