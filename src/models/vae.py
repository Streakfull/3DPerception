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
        self.encoder_channels = configs["auto_encoder_networks"]["out_channels"]
        self.conv_mu = nn.Conv3d(
            in_channels=self.encoder_channels, out_channels=self.encoder_channels, kernel_size=3, padding=1)
        if (self.is_vae):
            self.conv_logvar = nn.Conv3d(
                in_channels=self.encoder_channels, out_channels=self.encoder_channels, kernel_size=3, padding=1)

        self.kl = KLDivergence()

    @property
    def is_vae(self):
        return self.kl_weight > 0

    def forward(self, x):
        self.target = x
        x = self.encoder(x)

        self.mu = self.conv_mu(x)
        if (self.is_vae):
            self.logvar = self.conv_logvar(x)
            self.mu = rearrange(self.mu, 'b ch w h l -> b (ch w h l)')
            self.logvar = rearrange(self.logvar, 'b ch w h l -> b (ch w h l)')
            z = self._reparameterize(self.mu, self.logvar)
            z = rearrange(z, 'b (ch w h l) -> b ch w h l',
                          ch=self.encoder_channels, w=8, h=8, l=8)
        else:
            z = self.mu

        out = self.decoder(z)
        self.predictions = out
        return out

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def set_loss(self):
        self.reconst_loss = self.criterion(self.predictions, self.target)
        if (self.is_vae):
            self.kl_loss = self.kl(self.mu, self.logvar)
        else:
            self.kl_loss = torch.tensor(0.0)
        self.loss = (self.reconst_weight*self.reconst_loss) + \
            (self.kl_weight*self.kl_loss)

    def get_metrics(self):
        return {'loss': self.loss.data, 'l2': self.reconst_loss.data, 'kl': self.kl_loss.data}

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
        init_weights(self.conv_mu, init_type=init_type, gain=gain)
        if (self.is_vae):
            init_weights(self.conv_logvar, init_type=init_type, gain=gain)
        # init_type(self.decode_conv, init_type=init_type, gain=gain)
        # init_weights(self.mu_linear, init_type=init_type, gain=gain)
        # init_weights(self.logvar_linear, init_type=init_type, gain=gain)
        # init_weights(self.decode_linear, init_type=init_type, gain=gain)

    def sample(self, n_samples=1, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(
                size=(n_samples, self.configs["latent_space_size"])).to(device=device)
           # z = self.decode_linear(z)
            z = rearrange(z, 'b (ch w h l) -> b ch w h l',
                          ch=32, w=8, h=8, l=8)
           # z = self.decode_conv(z)
            out = self.decoder(z)
            return out
