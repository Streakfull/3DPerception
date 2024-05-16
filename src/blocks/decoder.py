import torch
from src.blocks.attn_block import AttnBlock
from src.blocks.res_net import ResnetBlock
from src.blocks.block_utils import nonlinearity, Normalize
from src.blocks.upsample import Upsample
import numpy as np
from torch import nn


class Decoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ch_mult=(1, 2, 4, 8), num_res_blocks=1,
                 attn_resolutions=[8], dropout=0.0, resamp_with_conv=True,
                 resolution=1):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        # self.in_channels = in_channels
        self.in_channels = 4
        block_in = self.in_channels
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1, self.in_channels, curr_res, curr_res, curr_res)
        self.sigmoid = nn.Sigmoid()
        print("Decoding of shape {} = {} dimensions.".format(
            self.z_shape, np.prod(self.z_shape)))

        # z to block_in
        self.conv_in = torch.nn.Conv3d(self.in_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()

        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)

        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       dropout=dropout)
        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = self.in_channels*(max(ch_mult[i_level]-1, 1))
            # for i_block in range(self.num_res_blocks+1):
            # change this to align with encoder
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Dec has Attn at i_level, i_block: %d, %d' %
                          (i_level, i_block))
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order
        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(in_channels=block_in,
                                        out_channels=1,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # for i_block in range(self.num_res_blocks+1):
            for i_block in range(self.num_res_blocks):  # change this to align encoder
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # h = self.sigmoid(h)
        return h
