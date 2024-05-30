from src.models.base_model import BaseModel
import torch
from torch import nn
from src.losses.KL_divergence import KLDivergence
from torch import optim
from src.losses.build_loss import BuildLoss
from src.utils.model_utils import init_weights
from einops import rearrange


class VAE3D(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.base_kl_weight = configs['base_kl_weight']
       # self.base_reconst_weight = configs['base_reconst_weight']
        self.cycle_iter = configs['cycle_iter']
        self.stop_cycle_count = configs['stop_cycle_count']
        self.encoder_channels = configs["auto_encoder_networks"]["out_channels"]
        self.reconst_weight = configs['reconst_weight']
        self.configs = configs

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=0,
            stride=1
        )

        self.conv2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
            stride=2,


        )

        self.conv3 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=0,
            stride=1

        )

        self.conv4 = nn.Conv3d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            stride=2
        )

        self.conv5 = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1,
            stride=2
        )

        self.encoder = nn.Sequential(
            self.conv1,
            nn.ELU(),
            nn.BatchNorm3d(8),

            self.conv2,
            nn.ELU(),
            nn.BatchNorm3d(16),


            self.conv3,
            nn.ELU(),
            nn.BatchNorm3d(32),


            self.conv4,
            nn.ELU(),
            nn.BatchNorm3d(64),

            self.conv5,
            nn.ELU(),
            nn.BatchNorm3d(128)
        )

        self.out_encoder_channels = 128 * 8 * 8 * 8
        self.encfc1 = nn.Sequential(
            nn.Linear(in_features=self.out_encoder_channels, out_features=512),
            nn.ELU(),
            # nn.BatchNorm1d(512),
        )

        self.mu_fc = nn.Sequential(

            nn.Linear(in_features=512, out_features=256),
            # nn.BatchNorm1d(256),
        )

        self.logvar_fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            # nn.BatchNorm1d(256),
        )

        self.dec_fc1 = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ELU(),
            # nn.BatchNorm1d(512)
        )

        self.dec_conv1 = nn.ConvTranspose3d(
            in_channels=1,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.dec_conv2 = nn.ConvTranspose3d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=0
        )

        self.dec_conv3 = nn.ConvTranspose3d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.dec_conv4 = nn.ConvTranspose3d(
            in_channels=32,
            out_channels=16,
            kernel_size=4,
            stride=2,
            padding=1

        )

        self.dec_conv5 = nn.ConvTranspose3d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=2

        )

        self.dec_conv6 = nn.ConvTranspose3d(
            in_channels=8,
            out_channels=1,
            kernel_size=4,
            stride=2,
            padding=1

        )

        self.decoder = nn.Sequential(
            self.dec_conv1,
            nn.ELU(),
            nn.BatchNorm3d(128),

            self.dec_conv2,
            nn.ELU(),
            nn.BatchNorm3d(64),


            self.dec_conv3,
            nn.ELU(),
            nn.BatchNorm3d(32),

            self.dec_conv4,
            nn.ELU(),
            nn.BatchNorm3d(16),

            self.dec_conv5,
            nn.ELU(),
            nn.BatchNorm3d(8),

            self.dec_conv6,
            nn.ELU(),
            nn.BatchNorm3d(1)
        )

        self.kl = KLDivergence()
        self.criterion = BuildLoss(configs).get_loss()
        self.optimizer = optim.Adam(
            params=self.parameters(), lr=configs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])

        self.norm_out_2 = nn.BatchNorm3d(1)
        self.sigmoid = nn.Sigmoid()

    @ property
    def is_vae(self):
        # return True
        return True
        return self.kl_weight > 0

    def forward(self, x):
        self.target = x
        x = self.encoder(x)
        x = x.flatten(1)

        x = self.encfc1(x)
        self.mu = self.mu_fc(x)
        self.logvar = self.logvar_fc(x)
        z = self._reparameterize(self.mu, self.logvar)
        x = self.dec_fc1(z)
        z = rearrange(x, 'bs (c l w h) -> bs c l w h', c=1, l=8, w=8, h=8)

        x = self.decoder(z)
        # x = self.norm_out_2(x)
        # x = self.sigmoid(x)*0.2
        self.x_reconst = x
        return x

    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + (eps * std)

    def init_weights(self):
        init_type = self.configs['weight_init']
        gain = self.configs['gain']
        init_weights(self, init_type=init_type, gain=gain)
        init_weights(self.encoder, init_type=init_type, gain=gain)
        init_weights(self.decoder, init_type=init_type, gain=gain)
        init_weights(self.encfc1, init_type=init_type, gain=gain)
        init_weights(self.mu_fc, init_type=init_type, gain=gain)
        init_weights(self.logvar_fc, init_type=init_type, gain=gain)
        init_weights(self.dec_fc1, init_type=init_type, gain=gain)
        init_weights(self.decoder, init_type=init_type, gain=gain)

    def set_iteration(self, iteration):
        self.iteration = iteration

    def sample(self, n_samples=1, device="cuda:0"):
        self.eval()
        with torch.no_grad():
            z = torch.randn(size=(n_samples, 256)).to(device=device)
            x = self.dec_fc1(z)
            z = rearrange(x, 'bs (c l w h) -> bs c l w h', c=1, l=8, w=8, h=8)
            x = self.decoder(z)
            return x

    def set_loss(self):
        self.reconst_loss = self.criterion(
            self.x_reconst, self.target)
        self.set_kl_weight()
        if (self.is_vae and self.kl_weight > 0):
            self.kl_loss = self.kl(self.mu, self.logvar)
        else:
            self.kl_loss = torch.tensor(0, device=self.x_reconst.device)

        self.loss = (self.reconst_weight*self.reconst_loss) + \
            (self.kl_weight*self.kl_loss)

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
        self.kl_weight = self.base_kl_weight * self.kl_weight

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def inference(self, x):
        self.eval()
        x = self.forward(x)
        return x

    def get_metrics(self):
        return {'loss': self.loss.data, 'l2': self.reconst_loss.data, 'kl': self.kl_loss.data, 'signedIou': 0}

    def get_batch_input(self, x):
        return x['sdf']

    def prepare_visuals(self):
        visuals = {
            "reconstructions": self.x_reconst,
            "target": self.target,
            "samples": self.sample(n_samples=self.x_reconst.shape[0]),

        }
        return visuals
