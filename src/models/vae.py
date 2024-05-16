from src.models.auto_encoder import AutoEncoder
from torch import nn
from einops import rearrange
import torch
from src.losses.KL_divergence import KLDivergence
from src.utils.model_utils import init_weights


class VAE(AutoEncoder):
    def __init__(self, configs):
        super().__init__(configs)
        self.kl_weight = configs['kl_weight']
        self.reconst_weight = configs['reconst_weight']
        self.mu_linear = nn.Linear(
            in_features=configs['encoder_out'], out_features=configs["latent_space_size"])

        # self.batch_norm_mu = nn.BatchNorm1d(
        #     num_features=configs["latent_space_size"])
        self.logvar_linear = nn.Linear(
            in_features=configs['encoder_out'], out_features=configs["latent_space_size"])
        # self.batch_norm_logvar = nn.BatchNorm1d(
        #     num_features=configs["latent_space_size"])
        self.decode_linear = nn.Linear(
            in_features=configs["latent_space_size"],
            out_features=configs['encoder_out'])

        self.conv_mu = nn.Conv3d(in_channels=64, out_channels=2, kernel_size=1)
        self.conv_logvar = nn.Conv3d(
            in_channels=64, out_channels=2, kernel_size=1)

        self.kl = KLDivergence()

    def forward(self, x):
        self.target = x
        x = self.encoder(x)

        self.mu = self.conv_mu(x)
        self.logvar = self.conv_logvar(x)
        self.mu = rearrange(self.mu, 'b ch w h l -> b (ch w h l)')
        self.logvar = rearrange(self.logvar, 'b ch w h l -> b (ch w h l)')
        z = self._reparameterize(self.mu, self.logvar)
        z = self.decode_linear(z)
        z = rearrange(z, 'b (ch w h l) -> b ch w h l', ch=1, w=8, h=8, l=8)
        out = self.decoder(z)
        self.predictions = out
        return out

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def set_loss(self):
        self.reconst_loss = self.reconst_weight * \
            self.criterion(self.predictions, self.target)
        self.kl_loss = self.kl_weight * self.kl(self.mu, self.logvar)
        self.loss = self.reconst_loss + self.kl_loss

    def get_metrics(self):
        return {'loss': self.loss.data, 'l2': self.reconst_loss.data / self.reconst_weight, 'kl': self.kl_loss.data / self.kl_weight}

    def calculate_additional_metrics(self):
        metrics = {}
        for metric in self.metrics:
            value = metric[1].calc_batch(self.predictions, self.target)
            metrics[metric[0]] = value
        return metrics

    def init_weights(self):
        super().init_weights()
        init_type = self.configs['weight_init']
        gain = self.configs['gain']
        init_weights(self.mu_linear, init_type=init_type, gain=gain)
        init_weights(self.logvar_linear, init_type=init_type, gain=gain)
        init_weights(self.decode_linear, init_type=init_type, gain=gain)

    def sample(self, n_samples=1, deivce="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(
                size=(n_samples, self.configs["latent_space_size"])).to(device=deivce)
            z = self.decode_linear(z)
            z = rearrange(z, 'b (ch w h l) -> b ch w h l', ch=1, w=8, h=8, l=8)
            out = self.decoder(z)
            return out
