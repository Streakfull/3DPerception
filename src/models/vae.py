from src.models.auto_encoder import AutoEncoder
from torch import nn
from einops import rearrange
import torch
from src.losses.KL_divergence import KLDivergence
from src.utils.model_utils import init_weights
from src.blocks.block_utils import Normalize, nonlinearity
from src.metrics.signed_iou import SignedIou
from torch import optim
import numpy as np
from src.blocks.block_utils import Normalize, nonlinearity


class VAE(AutoEncoder):
    def __init__(self, configs):
        super().__init__(configs)
        self.base_kl_weight = configs['base_kl_weight']
        self.cycle_iter = configs['cycle_iter']
        self.stop_cycle_count = configs['stop_cycle_count']
        self.encoder_channels = configs["auto_encoder_networks"]["out_channels"]
        self.reconst_weight = configs['reconst_weight']
        self.z_dim = configs['z_dim']
        self.use_cycles = configs['use_cycles']
        self.conv_in = nn.Conv3d(
            in_channels=self.encoder.out_channels, out_channels=2*self.z_dim, kernel_size=4)
        self.dec_in = nn.Linear(in_features=self.z_dim,
                                out_features=self.encoder_channels*64)
        self.norm_in_encoder = Normalize(self.encoder.out_channels)
        self.optimizer = optim.Adam(
            params=self.parameters(), lr=configs["lr"], betas=(0.5, 0.9))
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])

        self.kl = KLDivergence()

    def forward(self, x):
        self.target = x
        x = self.encoder(x)
        x = self.norm_in_encoder(x)
        self.mu, self.logvar = torch.chunk(self.conv_in(x), chunks=2, dim=1)
        if (self.training):
            z = self._reparameterize(self.mu, self.logvar)
        else:
            self.logvar = torch.zeros_like(self.mu, device=self.mu.device)
            z = self.mu

        x = self.decode(z)
        self.predictions = x
        return x

    def decode(self, z):
        z = z.flatten(1)
        z = self.dec_in(z)
        z = rearrange(z, 'bs (ch l w h)->bs ch l w h',
                      ch=self.encoder_channels, l=4, w=4, h=4)
        z = nonlinearity(z)
        x = self.decoder(z)
        return x

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def set_loss(self):
        self.reconst_loss = self.criterion(
            self.predictions, self.target)
        self.set_kl_weight()
        self.kl_loss = self.kl(self.mu, self.logvar)
        self.loss = (self.reconst_weight*self.reconst_loss) + \
            (self.kl_weight*self.kl_loss)

    def set_kl_weight(self):
        if (not self.use_cycles):
            self.kl_weight = self.base_kl_weight
            return
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
        self.kl_weight = self.base_kl_weight * self.kl_weight

    def get_metrics(self):
        return {'loss': self.loss.data, 'l1': self.reconst_loss.detach(), 'kl': self.kl_loss.data, 'kl_weight': self.kl_weight}

    def calculate_additional_metrics(self):
        metrics = {}
        for metric in self.metrics:
            value = metric[1].calc_batch(self.predictions, self.target)
            metrics[metric[0]] = value
        return metrics

    def init_weights(self):
        return

    def sample(self, n_samples=1, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(size=(n_samples, self.z_dim, 1, 1, 1)).to(
                device=device)
            out = self.decode(z)
            return out, z

    def sample_uniform(self, n_samples=1, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.rand(size=(n_samples, self.z_dim, 1, 1, 1)
                           ).to(device=device)
            out = self.decode(z)
            return out, z

    def set_iteration(self, iteration):
        self.iteration = iteration

    def prepare_visuals(self):
        visuals = {
            "reconstructions": self.predictions,
            "target": self.target,
            "samples": self.sample(n_samples=8)[0],
            # "samples_uniform": self.sample_uniform(n_samples=16)[0],

        }
        return visuals

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()
