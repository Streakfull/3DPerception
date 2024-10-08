from torch import nn
from src.losses.VQ_loss import VQLoss
from src.blocks.quantizer import VectorQuantizer
from src.blocks.encoder import Encoder
from src.models.base_model import BaseModel
from src.blocks.decoder import Decoder
from src.losses.build_loss import BuildLoss
from src.utils.model_utils import init_weights
from einops import rearrange
from torch import nn, optim
import torch


class PVQVAE(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.encoder = Encoder(**(configs["auto_encoder_networks"]))
        decoder_config = configs["auto_encoder_networks"]
        decoder_config["in_channels"] = self.encoder.out_channels
        self.decoder = Decoder(**decoder_config)
        self.criterion = BuildLoss(configs).get_loss()
        self.n_embed = configs["n_embed"]
        self.embed_dim = configs["embed_dim"]
        self.n_down = len(configs['auto_encoder_networks']['ch_mult'])-1
        self.quantize = VectorQuantizer(
            n_e=self.n_embed, e_dim=self.embed_dim, beta=1.0)
        self.configs = configs
        # self.cur_bs = 8
        self.quant_conv = nn.Conv3d(
            in_channels=self.encoder.out_channels, out_channels=self.embed_dim, kernel_size=1)

        self.post_quant_conv = nn.Conv3d(
            in_channels=self.embed_dim, out_channels=self.encoder.out_channels, kernel_size=1)

        self.set_metrics()
        self.optimizer = optim.Adam(
            self.parameters(), lr=configs['lr'], betas=(0.5, 0.9))
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        self.criterion = VQLoss()
        self.resolution = configs["auto_encoder_networks"]["resolution"]
        self.cur_bs = 4
        # setup hyper-params
        nC = self.resolution
        self.cube_size = 2 ** self.n_down  # patch_size
        self.stride = self.cube_size
        self.ncubes_per_dim = nC // self.cube_size
        assert nC == 64, 'right now, only trained with sdf resolution = 64'
        assert (nC % self.cube_size) == 0, 'nC should be divisable by cube_size'

    def init_weights(self):
        super().init_weights()
        init_type = self.configs['weight_init']
        gain = self.configs['gain']

        init_weights(self.encoder, init_type, gain)
        init_weights(self.decoder, init_type, gain)
        init_weights(self.quant_conv, init_type, gain)
        init_weights(self.post_quant_conv, init_type, gain)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h, is_voxel=True)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_from_quant(self, quant_code):
        z_q = self.quantize.embedding(quant_code)
        z_q = rearrange(z_q, 'bs d w h ch -> bs ch d w h')
        z_q = self.decode(z_q)
        return z_q

    def decode_enc_indices(self, enc_indices, z_spatial_dim=8):

        # for transformer
        enc_indices = rearrange(enc_indices, 't bs -> (bs t)')
        z_q = self.quantize.embedding(enc_indices)  # (bs t) zd
        z_q = rearrange(z_q, '(bs d1 d2 d3) zd -> bs zd d1 d2 d3',
                        d1=z_spatial_dim, d2=z_spatial_dim, d3=z_spatial_dim)
        dec = self.decode(z_q)
        return dec

    @staticmethod
    # def unfold_to_cubes(self, x, cube_size=8, stride=8):
    def unfold_to_cubes(x, cube_size=8, stride=8):
        """ 
            assume x.shape: b, c, d, h, w 
            return: x_cubes: (b cubes)
        """
        x_cubes = x.unfold(2, cube_size, stride).unfold(
            3, cube_size, stride).unfold(4, cube_size, stride)
        x_cubes = rearrange(
            x_cubes, 'b c p1 p2 p3 d h w -> b c (p1 p2 p3) d h w')
        x_cubes = rearrange(x_cubes, 'b c p d h w -> (b p) c d h w')

        return x_cubes

    @staticmethod
    # def fold_to_voxels(self, x_cubes, batch_size, ncubes_per_dim):
    def fold_to_voxels(x_cubes, batch_size, ncubes_per_dim):
        x = rearrange(x_cubes, '(b p) c d h w -> b p c d h w', b=batch_size)
        x = rearrange(x, 'b (p1 p2 p3) c d h w -> b c (p1 d) (p2 h) (p3 w)',
                      p1=ncubes_per_dim, p2=ncubes_per_dim, p3=ncubes_per_dim)
        return x

    def get_batch_input(self, x):
        x = x['sdf']
        self.x = x
        self.cur_bs = x.shape[0]
        return self.x

    def forward(self, x):
        self.train()
        self.x_cubes = self.unfold_to_cubes(x, self.cube_size, self.stride)
        # qloss: codebook loss
        self.zq_cubes, self.qloss, _ = self.encode(
            self.x_cubes)  # zq_cubes: ncubes X zdim X 1 X 1 X 1

        # zq_voxels: bs X zdim X ncubes_per_dim X ncubes_per_dim X ncubes_per_dim
        self.zq_voxels = self.fold_to_voxels(
            self.zq_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim)

        self.x_recon = self.decode(self.zq_voxels)
        return self.x_recon

    def inference(self, data):
        self.eval()
        # make sure it has the same name as forward
        with torch.no_grad():
            self.x_cubes = self.unfold_to_cubes(
                data, self.cube_size, self.stride)
            self.zq_cubes, qloss, self.info = self.encode(self.x_cubes)
            self.qloss = qloss
            self.zq_voxels = self.fold_to_voxels(
                self.zq_cubes, batch_size=self.cur_bs, ncubes_per_dim=self.ncubes_per_dim)
            self.x_recon = self.decode(self.zq_voxels)
            return self.x_recon

    def set_loss(self):
        loss_dict = self.criterion(
            self.qloss, self.x_recon, self.x, last_layer=self.get_last_layer(), global_step=self.iteration)

        self.loss = loss_dict["loss"]
        self.reconst_loss = loss_dict["l1"]
        self.codebook_loss = loss_dict["codebook"]
        self.p_loss = loss_dict["p"]

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def get_metrics(self, apply_additional_metrics=False):
        return {'loss': self.loss.data, 'codebook': self.codebook_loss.data, 'l1': self.reconst_loss}

    def calculate_additional_metrics(self):
        metrics = {}
        for metric in self.metrics:
            value = metric[1].calc_batch(self.x_recon, self.x)
            metrics[metric[0]] = value
        return metrics

    def prepare_visuals(self):
        visuals = {
            "reconstructions": self.x_recon,
            "target": self.x,
        }
        return visuals

    def set_iteration(self, iteration):
        self.iteration = iteration

    def get_codebook_weight(self):
        ret = self.quantize.embedding.cpu().state_dict()
        self.quantize.embedding.cuda()
        return ret

    def decode_enc_idices(self, enc_indices, z_spatial_dim=8):

        # for transformer
        enc_indices = rearrange(enc_indices, 't bs -> (bs t)')
        z_q = self.quantize.embedding(enc_indices)  # (bs t) zd
        z_q = rearrange(z_q, '(bs d1 d2 d3) zd -> bs zd d1 d2 d3',
                        d1=z_spatial_dim, d2=z_spatial_dim, d3=z_spatial_dim)
        dec = self.decode(z_q)
        return dec

    def get_last_layer(self):
        return self.decoder.conv_out.weight
