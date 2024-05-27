
import math
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder, LayerNorm
from einops import rearrange, repeat

from src.blocks.transformer.pos_embedding import PEPixelTransformer


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.configs = config
        self.codebook_size = config['pvqvae']['n_embed']
        vq_embed_dim = config['pvqvae']['embed_dim']

        p_encoding_conf = config['p_encoding']
        pos_embed_dim = p_encoding_conf['pos_embed_dim']

        n_tokens = config['n_tokens']
        tf_embed_dim = config['embed_dim']
        n_head = config['n_head']
        n_layers_enc = config['n_layers_enc']
        d_mlp = config['d_mlp']
        dropout = config['dropout']

        # Start Token
        self.embedding_start = nn.Embedding(1, vq_embed_dim)
        self.codebook = nn.Embedding(self.codebook_size, vq_embed_dim)

        self.pos_embedding = PEPixelTransformer(pe_conf=p_encoding_conf)
        self.fuse_linear = nn.Linear(
            in_features=vq_embed_dim+pos_embed_dim+pos_embed_dim, out_features=tf_embed_dim)

        encoder_layer = TransformerEncoderLayer(
            d_model=tf_embed_dim, nhead=n_head, dim_feedforward=d_mlp, dropout=dropout, activation='relu')

        encoder_norm = LayerNorm(tf_embed_dim)

        self.encoder = TransformerEncoder(
            encoder_layer, n_layers_enc, encoder_norm)

        self.dec_linear = nn.Linear(tf_embed_dim, n_tokens)
        self.d_tf = tf_embed_dim
        self._init_weights()

    def _init_weights(self) -> None:
        """initialize the weights of params."""

        _init_range = 0.1

        self.embedding_start.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size)
        self.codebook.weight.data.uniform_(
            -1.0 / self.codebook_size, 1.0 / self.codebook_size)

        self.fuse_linear.bias.data.normal_(0, 0.02)
        self.fuse_linear.weight.data.normal_(0, 0.02)

        self.dec_linear.bias.data.normal_(0, 0.02)
        self.dec_linear.weight.data.normal_(0, 0.02)

    def generate_square_subsequent_mask(self, sz, device):
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(device)

    def generate_square_id_mask(self, sz, device):
        mask = torch.eye(sz)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))
        mask = mask.to(device)
        return mask

    def forward_transformer(self, src, src_mask=None):
        output = self.encoder(src, mask=src_mask)
        # output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output

    def forward(self, inp, inp_posn, tgt_posn):
        """ Here we will have the full sequence of inp """
        device = inp.get_device()
        seq_len, bs = inp.shape[:2]
        tgt_len = tgt_posn.shape[0]

        # token embedding
        sos = inp[:1, :]
        inp_tokens = inp[1:, :]
        inp_val = torch.cat([self.embedding_start(sos), self.codebook(
            inp_tokens)], dim=0) * math.sqrt(self.d_tf)
        inp_posn = repeat(self.pos_embedding(inp_posn),
                          't pos_d -> t bs pos_d', bs=bs)
        tgt_posn = repeat(self.pos_embedding(tgt_posn),
                          't pos_d -> t bs pos_d', bs=bs)

        inp = torch.cat([inp_val, inp_posn, tgt_posn], dim=-1)

        # fusion
        inp = rearrange(inp, 't bs d -> (t bs) d')
        inp = rearrange(self.fuse_linear(
            inp), '(t bs) d -> t bs d', t=seq_len, bs=bs)

        src_mask = self.generate_square_subsequent_mask(seq_len, device)
        outp = self.forward_transformer(inp, src_mask=src_mask)
        outp = self.dec_linear(outp)

        return outp
