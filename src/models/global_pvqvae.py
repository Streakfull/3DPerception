from torch import nn
from src.losses.VQ_loss import VQLoss
from src.losses.VQLossDisc import VQLossDisc
from src.blocks.quantizer import VectorQuantizer
from src.blocks.encoder import Encoder
from src.models.base_model import BaseModel
from src.blocks.decoder import Decoder
from src.losses.build_loss import BuildLoss
from src.utils.model_utils import init_weights
from einops import rearrange
from torch import nn, optim
import torch
from termcolor import cprint


class GlobalPVQVAE(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.encoder = Encoder(**(configs["auto_encoder_networks"]))
        decoder_config = configs["auto_encoder_networks"]
        decoder_config["in_channels"] = self.encoder.out_channels
        self.decoder = Decoder(**decoder_config)
        self.n_embed = configs["n_embed"]
        self.embed_dim = configs["embed_dim"]
        self.n_down = len(configs['auto_encoder_networks']['ch_mult'])-1
        self.use_disc = configs["use_disc"]
        self.quantize = VectorQuantizer(
            n_e=self.n_embed, e_dim=self.embed_dim, beta=1.0)
        self.configs = configs
        # self.cur_bs = 8
        self.quant_conv = nn.Conv3d(
            in_channels=self.encoder.out_channels, out_channels=self.embed_dim, kernel_size=1)

        self.post_quant_conv = nn.Conv3d(
            in_channels=self.embed_dim, out_channels=self.encoder.out_channels, kernel_size=1)

        self.set_metrics()
        if (self.use_disc):
            self.criterion = VQLossDisc(
                vgg_checkpoint=configs['vgg_ckpt'], perceptual_weight=configs[
                    "perceptual_weight"], disc_weight=self.configs["disc_weight"],
                disc_start=self.configs['disc_start']
            )
        else:
            self.criterion = VQLoss(
                vgg_checkpoint=configs['vgg_ckpt'], perceptual_weight=configs["perceptual_weight"])
        self.resolution = configs["auto_encoder_networks"]["resolution"]
        self.resolution = configs["auto_encoder_networks"]["resolution"]

        if (self.use_disc):
            self.opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                           list(self.decoder.parameters()) +
                                           list(self.quantize.parameters()) +
                                           list(self.quant_conv.parameters()) +
                                           list(
                                               self.post_quant_conv.parameters()),
                                           lr=self.configs["lr"], betas=(0.5, 0.9))

            self.scheduler = optim.lr_scheduler.StepLR(
                self.opt_ae, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])

            self.opt_disc = torch.optim.Adam(self.criterion.discriminator.parameters(),
                                             lr=configs['disc_lr'], betas=(0.5, 0.9))
            self.schedulerDisc = optim.lr_scheduler.StepLR(
                self.opt_disc, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        else:
            self.optimizer = optim.Adam(
                [p for p in self.parameters() if p.requires_grad == True], lr=configs['lr'], betas=(0.5, 0.9))
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])

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

    def get_batch_input(self, x):
        x = x['sdf']
        self.x = x
        self.cur_bs = x.shape[0]
        return self.x

    def forward(self, x):
        self.train()
        self.zq_cubes, self.qloss, _ = self.encode(
            x)
        self.x_recon = self.decode(self.zq_cubes)
        return self.x_recon

    def inference(self, data):
        self.eval()
        # make sure it has the same name as forward
        with torch.no_grad():

            self.zq_cubes, qloss, self.info = self.encode(data)
            self.qloss = qloss
            self.x_recon = self.decode(self.zq_cubes)
            return self.x_recon

    def set_loss(self):
        loss_dict = self.criterion(
            self.qloss, self.x_recon, self.x, last_layer=self.get_last_layer(), global_step=self.iteration)

        self.loss = loss_dict["loss"]
        self.reconst_loss = loss_dict["l1"]
        self.codebook_loss = loss_dict["codebook"]
        self.p_loss = loss_dict["p"]
        if (self.use_disc):
            self.disc_factor = loss_dict["disc_factor"]
            self.g_loss = loss_dict["g_loss"]
            self.d_weight = loss_dict["d_weight"]

            d_loss, loss_dict = self.criterion(
                self.qloss, self.x_recon, self.x, last_layer=self.get_last_layer(), global_step=self.iteration, optimizer_idx=1)
            self.d_loss = d_loss
            self.d_loss_copy = loss_dict['disc_loss']
            self.logits_real = loss_dict['logits_real']
            self.logits_fake = loss_dict['logits_fake']

    def backward(self):
        self.set_loss()
        self.loss.backward()
        self.d_loss.backward()

    def step(self, x):
        if (self.use_disc):
            self.step_disc(x)
        else:
            self.step_vanilla(x)

    def step_disc(self, x):
        self.train()
        self.opt_ae.zero_grad()
        self.opt_disc.zero_grad()
        x = self.forward(x)
        self.backward()
        self.opt_ae.step()
        self.opt_disc.step()

    def step_vanilla(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def get_metrics(self, apply_additional_metrics=False):
        if (self.use_disc):
            return {'loss': self.loss.data,
                    'codebook': self.codebook_loss.data,
                    'l1': self.reconst_loss,
                    'p': self.p_loss,
                    'disc_factor': self.disc_factor,
                    'g_loss': self.g_loss,
                    'd_weight': self.d_weight,
                    'disc_loss': self.d_loss_copy,
                    'logits_real': self.logits_real,
                    'logits_fake': self.logits_fake}
        else:
            return {'loss': self.loss.data, 'codebook': self.codebook_loss.data, 'l1': self.reconst_loss, 'p': self.p_loss}

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

    def load_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path)
        state_dict_copy = {}
        for key in state_dict.keys():
            # if "criterion" in key:
            #     continue
            state_dict_copy[key] = state_dict[key]

        self.load_state_dict(state_dict_copy, strict=False)
        cprint(f"Model loaded from {ckpt_path}")

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def update_lr(self):
        if (self.use_disc):
            self.scheduler.step()
            lr = self.opt_ae.param_groups[0]['lr']
            cprint('[*] learning rate for opt_ae = %.7f' % lr, "yellow")

            self.schedulerDisc.step()
            lr = self.opt_disc.param_groups[0]['lr']
            cprint('[*] learning rate for opt_disc = %.7f' % lr, "yellow")
        else:
            self.scheduler.step()
            lr = self.optimizer.param_groups[0]['lr']
            cprint('[*] learning rate = %.7f' % lr, "yellow")
