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
from torch.distributions import MultivariateNormal


class VAE(AutoEncoder):
    def __init__(self, configs):
        super().__init__(configs)
        self.base_kl_weight = configs['base_kl_weight']
       # self.base_reconst_weight = configs['base_reconst_weight']
        self.cycle_iter = configs['cycle_iter']
        self.stop_cycle_count = configs['stop_cycle_count']
        self.encoder_channels = configs["auto_encoder_networks"]["out_channels"]
        self.reconst_weight = configs['reconst_weight']
        self.embed_dim = 512
        self.epoch = 0
        self.conv_in = nn.Conv3d(
            in_channels=self.encoder.out_channels, out_channels=64, kernel_size=1)

        self.dec_in = nn.Conv3d(
            in_channels=32, out_channels=self.encoder.out_channels, kernel_size=1)

        # self.norm_mu = nn.LayerNorm(2048)
        # self.norm_log_var = nn.LayerNorm(2048)
        self.optimizer = optim.Adam(
            params=self.parameters(), lr=configs["lr"], betas=(0.5, 0.9))
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])

        self.kl = KLDivergence()
        self.iter_per_epoch = 1

    @ property
    def is_vae(self):
        # return True
        return True
        return self.kl_weight > 0

    def forward(self, x):
        self.target = x
        x = self.encoder(x)
        x = self.conv_in(x)
        self.mu, self.logvar = torch.chunk(x, chunks=2, dim=1)

        if (self.is_vae):
            if (self.training):
                z = self._reparameterize(self.mu, self.logvar)
            else:
                self.logvar = torch.zeros_like(self.mu, device=self.mu.device)
                z = self.mu

        else:
            z = self.mu

        x = self.decode(z)
        self.predictions = x
        return x

    def decode(self, z):
        # z = self.z_linear_out(z)
        # z = rearrange(
        #     z, "bs (ch l w h)-> bs ch l w h", ch=4, l=8, w=8, h=8
        # )
        z = self.dec_in(z)
        z = self.decoder(z)
        return z

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def set_loss(self):
        self.reconst_loss = self.criterion(
            self.predictions, self.target)
        self.set_kl_weight()
        if (self.is_vae):
            self.kl_loss = self.kl(self.mu, self.logvar)
        else:
            self.kl_loss = torch.tensor(0, device=self.predictions.device)
        self.loss = (self.reconst_weight*self.reconst_loss) + \
            (self.kl_weight*self.kl_loss)

        # self.loss = (self.reconst_weight*self.reconst_loss)
        # self.loss = (self.reconst_weight * self.reconst_loss) + \
        #     (self.kl_weight*self.kl_loss)

        self.signedIou = 0

    # def set_kl_weight(self):
    #     # 14999
    #     div = self.iteration / self.cycle_iter
    #     current_iteration = 0
    #     if (div < 0):
    #         current_iteration = self.iteration
    #     if (div > 0):
    #         current_iteration = self.iteration - \
    #             (self.iteration//self.cycle_iter)*self.cycle_iter
    #     self.kl_weight = min(current_iteration/(self.cycle_iter*0.5), 1)
    #     if (self.iteration//self.stop_cycle_count) >= 1:
    #         self.kl_weight = 1
    #     # self.kl_weight /= (self.encoder_channels * 8 * 8 * 8)
    #     self.kl_weight = (self.base_kl_weight *
    #                       self.kl_weight) + self.base_kl_weight

    def set_kl_weight(self):
        # 14999
        self.kl_weight = self.base_kl_weight
        return
        self.warmup = 1
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
        # self.kl_weight /= (self.encoder_channels * 8 * 8 * 8)
        # adj_iter = self.iteration - (self.warmup *
        #                              self.iter_per_epoch)
        adj_iter = self.iteration
        warm_up_cycles = (self.warmup * self.iter_per_epoch)//self.cycle_iter
        current_cycle = (max(adj_iter, 1)//self.cycle_iter)/(5)
        current_cycle = max(current_cycle, 1)
        # current_cycle = (self.epoch//self.warmup) + 1
        self.kl_weight = (self.base_kl_weight *
                          self.kl_weight) + (int(current_cycle) * self.base_kl_weight)
        if (self.epoch < self.warmup):
            self.kl_weight = self.base_kl_weight

    def set_iter_per_epoch(self, iter):
        self.iter_per_epoch = iter

    def get_metrics(self):
        return {'loss': self.loss.data, 'l1': self.reconst_loss.detach().mean(), 'kl': self.kl_loss.data, 'kl_weight': self.kl_weight}

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
        init_weights(self.conv_in, init_type=init_type, gain=gain)
        # init_weights(self.linear_logvar, init_type=init_type, gain=gain)
        # init_weights(self.linear_mu, init_type=init_type, gain=gain)
        # init_weights(self.linear_out, init_type=init_type, gain=gain)

    def sample(self, n_samples=1, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(size=(n_samples, 32, 8, 8, 8)).to(device=device)
           # z = self.norm_mu(z.flatten(1))
            # z = rearrange(z, 'bs (ch w l h) ->bs ch w l h',
            #               ch=4, w=8, l=8, h=8)
            out = self.decode(z)
            return out, z

    def sample_normal(self, n_samples=1, mu=0, std=0, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            # z = torch.randn(size=(n_samples, 256)).to(device=device)
            m = MultivariateNormal(mu, torch.diag(std))
            z = m.sample([n_samples])
           # z = self.norm_mu(z)
            z = rearrange(z, 'bs (ch w l h) ->bs ch w l h',
                          ch=4, w=8, l=8, h=8)

            out = self.decode(z)
            return out

    def sample_tanh(self, n_samples=1, mu=0, std=0, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(size=(n_samples, 8, 8, 8, 8)).to(device=device)
            out = self.decode(z)
            out = nn.functional.tanh(out)*0.2
            return out, z

    def set_iteration(self, iteration):
        self.iteration = iteration

    def set_epoch(self, epoch):
        self.epoch = epoch

    def prepare_visuals(self):
        visuals = {
            "reconstructions": self.predictions,
            "target": self.target,
            "samples": self.sample(n_samples=self.predictions.shape[0]*4)[0],
            # "tanhSamples": self.sample_tanh(n_samples=self.predictions.shape[0]*4)[0]


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
