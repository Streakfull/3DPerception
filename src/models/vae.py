from src.models.auto_encoder import AutoEncoder
from torch import nn
from einops import rearrange
import torch
from src.losses.KL_divergence import KLDivergence
from src.utils.model_utils import init_weights
from src.blocks.block_utils import Normalize, nonlinearity
from src.metrics.signed_iou import SignedIou


class VAE(AutoEncoder):
    def __init__(self, configs):
        super().__init__(configs)
        self.base_kl_weight = configs['base_kl_weight']
        self.stop_cycle_count = configs['stop_cycle_count']
        self.reconst_weight = configs['reconst_weight']
        self.cycle_iter = configs['cycle_iter']
        self.encoder_channels = configs["auto_encoder_networks"]["out_channels"]
        self.conv_mu = nn.Conv3d(
            in_channels=64, out_channels=self.encoder_channels, kernel_size=3, padding=1)
        # self.linear_mu = nn.Linear(
        # in_features=1024, out_features=512)
        if (self.is_vae):
            self.conv_logvar = nn.Conv3d(
                in_channels=64, out_channels=self.encoder_channels, kernel_size=3, padding=1)
            # self.linear_logvar = nn.Linear(
            # in_features=1024, out_features=512)

        self.kl = KLDivergence()

    @property
    def is_vae(self):
        return True
        return self.kl_weight > 0

    def forward(self, x):
        self.target = x
        x = self.encoder(x)

        self.mu = self.conv_mu(x)
        # self.mu = nonlinearity(self.mu)
        # self.mu = self.linear_mu(self.mu.flatten(1))
        # self.mu = self.norm_mu(self.mu)
        if (self.is_vae and self.training):
            self.logvar = self.conv_logvar(x)
            # self.logvar = nonlinearity(self.logvar)
            # self.logvar = self.linear_logvar(self.logvar.flatten(1))
            z = self._reparameterize(self.mu, self.logvar)
            # z = rearrange(z, 'bs (ch l w h) -> bs ch l w h',
            #    ch=1, l=8, w=8, h=8)

        else:
            z = self.mu
            # z = rearrange(z, 'bs (ch l w h) -> bs ch l w h',
            #    ch=1, l=8, w=8, h=8)

        x = self.decoder(z)
        self.predictions = x
        return x

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def set_loss(self):
        self.reconst_loss = self.criterion(self.predictions, self.target)
        self.set_kl_weight()
        if (self.is_vae):
            self.kl_loss = self.kl(self.mu, self.logvar)
        self.loss = (self.reconst_weight*self.reconst_loss) + \
            (self.kl_weight*self.kl_loss)
        self.signedIou = SignedIou().calc(self.predictions, self.target)

    def set_kl_weight(self):
        div = self.iteration / self.cycle_iter
        current_iteration = 0
        if (div < 0):
            current_iteration = self.iteration
        if (div > 0):
            current_iteration = self.iteration - \
                (self.iteration//self.cycle_iter)*self.cycle_iter
        self.kl_weight = min(current_iteration/(self.cycle_iter*0.5), 1)
        if (self.iteration//self.stop_cycle_count) >= 1:
            self.kl_weight = 1
        self.kl_weight /= (self.encoder_channels * 8 * 8 * 8)
        self.kl_weight = self.base_kl_weight * self.kl_weight

    def get_metrics(self):
        return {'loss': self.loss.data, 'l2': self.reconst_loss.data, 'kl': self.kl_loss.data, 'signedIou': self.signedIou}

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
       # init_weights(self.linear_mu, init_type=init_type, gain=gain)
        if (self.is_vae):
            init_weights(self.conv_logvar, init_type=init_type, gain=gain)
           # init_weights(self.linear_logvar, init_type=init_type, gain=gain)

    def sample(self, n_samples=1, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(
                size=(n_samples, self.encoder_channels, 8, 8, 8)).to(device=device)
            out = self.decoder(z)
            return out, z

    def set_iteration(self, iteration):
        self.iteration = iteration

    def prepare_visuals(self):
        visuals = {
            "reconstructions": self.predictions,
            "target": self.target,
            "samples": self.sample(n_samples=self.predictions.shape[0])[0],

        }
        return visuals
