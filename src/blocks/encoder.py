from torch import nn
import torch
from src.blocks.res_net import ResnetBlock
from src.blocks.attn_block import AttnBlock
from src.blocks.downsample import Downsample
from src.blocks.block_utils import Normalize, nonlinearity


class Encoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, ch_mult=(1, 2, 4, 8), num_res_blocks=1,
                 attn_resolutions=[8], dropout=0.0, resamp_with_conv=True,
                 resolution=1):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.l1_output = self.in_channels * 64
        self.out_channels = out_channels
        # downsampling
        self.norm_in = Normalize(in_channels=in_channels)
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.l1_output,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        curr_res = resolution
        in_ch_mult = (1,)+tuple(ch_mult)  # (1, 1, 2, 4,8)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = self.l1_output*in_ch_mult[i_level]
            block_out = self.l1_output*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    print('[*] Enc has Attn at i_level, i_block: %d, %d' %
                          (i_level, i_block))
                    attn.append(AttnBlock(block_in))

                down = nn.Module()
                down.block = block
                down.attn = attn
                if i_level != self.num_resolutions-1:
                    down.downsample = Downsample(block_in, resamp_with_conv)
                    curr_res = curr_res // 2
            self.down.append(down)

            # middle
            self.mid = nn.Module()
            self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)
            self.mid.attn_1 = AttnBlock(block_in)
            self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                           out_channels=block_in,
                                           dropout=dropout)

        self.norm_out = Normalize(block_in)
        # self.norm_out_2 = Normalize(self.out_channels)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_channels=self.out_channels,
                                        # out_channels=64,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

        # self.conv_mu = torch.nn.Conv3d(block_in,
        #                                out_channels=self.out_channels,
        #                                # out_channels=32,
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1)
        # self.conv_logvar = torch.nn.Conv3d(block_in,
        #                                    out_channels=self.out_channels,
        #                                    # out_channels=32,
        #                                    kernel_size=3,
        #                                    stride=1,
        #                                    padding=1)

        # self.norm_out_2 = Normalize(64)

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(
        #     x.shape[2], x.shape[3], self.resolution)

        # h = self.norm_in(x)
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                # h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].block[i_block](h)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # hs.append(h)
            if i_level != self.num_resolutions-1:
                # hs.append(self.down[i_level].downsample(hs[-1]))
                h = self.down[i_level].downsample(h)

        # middle
        # h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        # h = self.norm_out_2(h)
        # h = nonlinearity(h)

        return h
