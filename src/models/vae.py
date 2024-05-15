from src.models.auto_encoder import AutoEncoder
from torch import nn
from einops import rearrange
import torch


class VAE(AutoEncoder):
    def __init__(self, configs):
        super().__init__(configs)
        self.kl_weight = configs['kl_weight']
        self.reconst_weight = configs['reconst_weight']
        # self.mu_conv = torch.nn.Conv3d(
        #     in_channels=configs["auto_encoder_networks"]["out_channels"],
        #     out_channels=1,
        #     kernel_size=1
        # )
        self.mu_linear = nn.Linear(
            in_features=configs['encoder_out'], out_features=configs["latent_space_size"])
        self.logvar_linear = nn.Linear(
            in_features=configs['encoder_out'], out_features=configs["latent_space_size"])

        self.decode_linear = nn.Linear(
            in_features=configs["latent_space_size"],
            out_features=configs['encoder_out'])

    def forward(self, x):
        self.target = x
        x = self.encoder(x)
        # self.mu = self.mu_conv(x)
        x = rearrange(x, 'b ch w h l -> b (ch w h l)')
        self.mu = self.mu_linear(x)
        self.logvar = self.logvar_linear(x)
        z = self._reparameterize(self.mu, self.logvar)
        z = self.decode_linear(z)
        z = rearrange(z, 'b (ch w h l) -> b ch w h l', ch=1, w=8, h=8, l=8)
        out = self.decoder(z)
        return out

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
