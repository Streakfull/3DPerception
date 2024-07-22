from torch import nn
import torch
from src.models.discriminator import NLayerDiscriminator, weights_init
from src.losses.LPIPIS import LPIPS
import torch.nn.functional as F


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real, logits_fake):
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class VQLossDisc(nn.Module):
    def __init__(self, codebook_weight=1.0, vgg_checkpoint=" logs/VGG/trainFull/2024_07_01_17_50_17/checkpoints/epoch-36.ckpt", perceptual_weight=0, disc_weight=0.8,
                 disc_loss="hinge", disc_start=25001, disc_factor=1):
        super().__init__()
        self.codebook_weight = codebook_weight
        self.perceptual_weight = perceptual_weight
        self.discriminator_weight = disc_weight
        if (self.perceptual_weight > 0):
            self.LPIPIS = LPIPS(ckpt_path=vgg_checkpoint).eval()
        self.discriminator = NLayerDiscriminator(input_nc=1,
                                                 n_layers=3,
                                                 use_actnorm=False,
                                                 ndf=64
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_factor = disc_factor
        self.bce = nn.BCEWithLogitsLoss()

        if disc_loss == "hinge":
            self.d_loss = hinge_d_loss
        elif disc_loss == "vanilla":
            self.d_loss = vanilla_d_loss
        else:
            raise ValueError(f"Unknown GAN loss '{disc_loss}'.")

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(
                nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(
                g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, codebook_loss, reconstructions, inputs, optimizer_idx=0,
                global_step=0, last_layer=None, cond=None, split="train"):

        if (optimizer_idx == 0):
            return self.ae_loss(inputs, reconstructions, codebook_loss, last_layer, global_step)
        else:
            return self.disc_loss(inputs, reconstructions, global_step)

    def ae_loss(self, inputs, reconstructions, codebook_loss, last_layer, global_step):
        rec_loss = torch.abs(inputs.contiguous() -
                             reconstructions.contiguous())
        l1_copy = rec_loss.detach().mean()
        if (self.perceptual_weight > 0):
            p_loss = self.LPIPIS(reconstructions, inputs)
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor(0, dtype=rec_loss.dtype)

        nll_loss = torch.mean(rec_loss)

        logits_fake = self.discriminator(reconstructions.contiguous())
        g_loss = self.bce(logits_fake, torch.ones_like(
            logits_fake).to(logits_fake.device))

        try:
            d_weight = self.calculate_adaptive_weight(
                nll_loss, g_loss, last_layer=last_layer)
        except RuntimeError:
            assert not self.training
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.discriminator_iter_start)

        loss = nll_loss + self.codebook_weight * codebook_loss.mean() + d_weight * \
            disc_factor * g_loss

        return {"loss": loss, "l1": l1_copy, "codebook": codebook_loss, "p": torch.abs(p_loss.detach().mean())*10,
                "disc_factor": disc_factor,
                "d_weight": d_weight.detach(),
                "g_loss": g_loss.detach().mean()
                }

    def disc_loss(self, inputs, reconstructions, global_step):
        logits_real = self.discriminator(inputs.contiguous().detach())
        logits_fake = self.discriminator(reconstructions.contiguous().detach())

        disc_factor = adopt_weight(
            self.disc_factor, global_step, threshold=self.discriminator_iter_start)
        d_loss = self.d_loss(logits_real, logits_fake)
        d_loss_copy = d_loss.detach().mean()
        d_loss = disc_factor * d_loss

        log = {"disc_loss": d_loss_copy.clone().detach().mean(),
               "logits_real": logits_real.detach().mean(),
               "logits_fake": logits_fake.detach().mean()
               }
        return d_loss, log
