import torch
import torch.nn as nn
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
from src.models.base_model import BaseModel
from src.losses.build_loss import BuildLoss


class Encoder(torch.nn.Module):
    def __init__(self, in_channels=1, dim=64, out_conv_channels=512):
        super(Encoder, self).__init__()
        conv1_channels = int(out_conv_channels / 8)
        conv2_channels = int(out_conv_channels / 4)
        conv3_channels = int(out_conv_channels / 2)
        self.out_conv_channels = out_conv_channels
        self.out_dim = int(dim / 8)

        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels, out_channels=conv1_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv1_channels, out_channels=conv2_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv2_channels, out_channels=conv3_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv3_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(
                in_channels=conv3_channels, out_channels=out_conv_channels, kernel_size=4,
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(out_conv_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # self.out = nn.Sequential(
        #     nn.Linear(out_conv_channels * self.out_dim *
        #               self.out_dim * self.out_dim, 1),
        # )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

       # x = self.conv4(x)

        # Flatten and apply linear + sigmoid
        # x = x.view(-1, self.out_conv_channels *
        #            self.out_dim * self.out_dim * self.out_dim)
        x = x.flatten(1)
        # x = self.out(x)
        return x


class Decoder(torch.nn.Module):
    def __init__(self, in_channels=256, out_dim=64, out_channels=1, noise_dim=200, activation="tanh"):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.in_dim = int(out_dim / 8)
        conv1_out_channels = int(self.in_channels / 2.0)
        conv2_out_channels = int(conv1_out_channels / 2)
        conv3_out_channels = int(conv2_out_channels / 2)

        self.linear = torch.nn.Linear(
            noise_dim, in_channels * self.in_dim * self.in_dim * self.in_dim)

        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=(
                    4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv1_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=(
                    4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            nn.BatchNorm3d(conv2_out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=conv2_out_channels, out_channels=1, kernel_size=(
                    4, 4, 4),
                stride=2, padding=1, bias=False
            ),
            # nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        # self.conv4 = nn.Sequential(
        #     nn.ConvTranspose3d(
        #         in_channels=conv3_out_channels, out_channels=out_channels, kernel_size=(
        #             4, 4, 4),
        #         stride=2, padding=1, bias=False
        #     )
        # )
        if activation == "sigmoid":
            self.out = torch.nn.Sigmoid()
        else:
            self.out = torch.nn.Tanh()

    def project(self, x):
        """
        projects and reshapes latent vector to starting volume
        :param x: latent vector
        :return: starting volume
        """
        return x.view(-1, self.in_channels, self.in_dim, self.in_dim, self.in_dim)

    def forward(self, x):
        x = self.linear(x)
        x = self.project(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
       # return x
        return self.out(x)*0.2


class VAE(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.base_kl_weight = configs['base_kl_weight']
        self.configs = configs
       # self.base_reconst_weight = configs['base_reconst_weight']
        self.cycle_iter = configs['cycle_iter']
        self.stop_cycle_count = configs['stop_cycle_count']
        self.encoder_channels = configs["auto_encoder_networks"]["out_channels"]
        self.reconst_weight = configs['reconst_weight']
        self.embed_dim = 200
        # self.linear_in = nn.Linear(
        #     in_features=self.encoder_channels*512, out_features=512)
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.criterion = BuildLoss(configs).get_loss()
        self.set_metrics()
        self.linear_in = nn.Sequential(
            nn.Linear(256*8*8*8, out_features=512),
            # nn.Conv3d(in_channels=256, out_channels=1, kernel_size=1),
            nn.Flatten(start_dim=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True))
        self.posterior = nn.Linear(in_features=512, out_features=400)
       # self.single_channel =

        self.optimizer = optim.Adam(
            params=self.parameters(), lr=configs["lr"], betas=(0.5, 0.9))
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        self.kl = KLDivergence()

    def forward(self, x):
        self.target = x
        x = self.encoder(x)
        x = self.linear_in(x)
        x = self.posterior(x)
        self.mu, self.logvar = torch.chunk(x, dim=1, chunks=2)
        if (self.training):
            z = self._reparameterize(self.mu, self.logvar)
        else:
            self.logvar = torch.zeros_like(self.mu, device=self.mu.device)
            z = self.mu
       # z = self.norm_z(z.flatten(1))
        x = self.decode(z)
        self.predictions = x
        # import pdb
        # pdb.set_trace()
        return x

    def decode(self, z):
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
        # self.kl_loss = torch.tensor(0, device=self.predictions.device)

        self.loss = (self.reconst_weight*self.reconst_loss) + \
            (self.kl_weight*self.kl_loss)
        # self.loss = (self.reconst_weight*self.reconst_loss)
        # self.loss = (self.reconst_weight * self.reconst_loss) + \
        #     (self.kl_weight*self.kl_loss)

    def set_kl_weight(self):
        # 14999
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
        current_cycle = self.iteration//self.cycle_iter
        self.kl_weight = (self.base_kl_weight * self.kl_weight) + \
            (current_cycle*self.base_kl_weight) + 1.0e-8
        self.kl_weight = min(self.kl_weight, 1.0e-2)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_iter_per_epoch(self, iter):
        self.iter_per_epoch = iter

    def get_metrics(self):
        return {'loss': self.loss.data, 'l1': self.reconst_loss.detach(), 'kl': self.kl_loss.data, 'kl_weight': self.kl_weight, 'mu_mean': self.mu.detach().mean()}

    def calculate_additional_metrics(self):
        metrics = {}
        for metric in self.metrics:
            value = metric[1].calc_batch(self.predictions, self.target)
            metrics[metric[0]] = value
        return metrics

    def init_weights(self):
        return
        super().init_weights()
        init_type = self.configs['weight_init']
        gain = self.configs['gain']
        # init_weights(self.conv_in, init_type=init_type, gain=gain)
        # init_weights(self.linear_logvar, init_type=init_type, gain=gain)
        # init_weights(self.linear_mu, init_type=init_type, gain=gain)
        # init_weights(self.linear_out, init_type=init_type, gain=gain

    def sample(self, n_samples=1, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(size=(n_samples, 200)).to(device=device)
            out = self.decode(z)
            return out, z

    def set_iteration(self, iteration):
        self.iteration = iteration

    def prepare_visuals(self):
        visuals = {
            "reconstructions": self.predictions,
            "target": self.target,
            "samples": self.sample(n_samples=max(self.predictions.shape[0]*4, 16))[0],


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

    def get_batch_input(self, x):
        return x['sdf']

    def inference(self, x):
        self.eval()
        x = self.forward(x)
        return x
