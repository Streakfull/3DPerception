import yaml
import torch
from torch import nn, optim
from cprint import *
from tqdm.notebook import tqdm
import torch.nn.functional as F

from einops import rearrange
from src.models.base_model import BaseModel
from src.blocks.transformer.transformer import Transformer
from src.models.pvqvae import PVQVAE


class RandTransformer(BaseModel):

    def __init__(self, configs, configs_path="src/configs/global_configs.yaml"):
        super().__init__()
        self.configs = configs
        self.tf = Transformer(config=configs)
        with open(configs_path, "r") as in_file:
            self.global_configs = yaml.safe_load(in_file)
        vqvae_config = self.global_configs["model"]["pvqvae"]
        self.vqvae = PVQVAE(vqvae_config)
        self.vqvae.load_ckpt(configs['pvqvae']['ckpt_path'])
        self.vqvae.eval()

        n_embed = configs['pvqvae']['n_embed']
        embed_dim = configs['pvqvae']['embed_dim']

        self.tf.embedding_encoder = nn.Embedding(n_embed, embed_dim)
        self.tf.embedding_encoder.load_state_dict(
            self.vqvae.quantize.embedding.state_dict())
        self.tf.embedding_encoder.requires_grad = False

        self.criterion_ce = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            [p for p in self.tf.parameters() if p.requires_grad == True], lr=configs['lr'])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 30, 0.9)

        self.sos = 0
        self.counter = 0

        pe_conf = configs['p_encoding']
        self.grid_size = pe_conf['zq_dim']
        self.grid_table = self.init_grid(
            pos_dim=pe_conf['pos_dim'], zq_dim=self.grid_size)

    def init_grid(self, pos_dim=3, zq_dim=8):
        x = torch.linspace(-1, 1, zq_dim)
        y = torch.linspace(-1, 1, zq_dim)
        if pos_dim == 3:
            z = torch.linspace(-1, 1, zq_dim)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z)
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
            pos_sos = torch.tensor(
                [-1., -1., -1-2/zq_dim]).float().unsqueeze(0)
        else:
            grid_x, grid_y = torch.meshgrid(x, y)
            grid = torch.stack([grid_x, grid_y], dim=-1)
            pos_sos = torch.tensor([-1., -1-2/zq_dim]).float().unsqueeze(0)

        grid_table = grid.view(-1, pos_dim)
        grid_table = torch.cat([pos_sos, grid_table], dim=0)
        return grid_table

    def get_gen_order(self, sz, device):
        # return torch.randperm(sz).to(device)
        return torch.randperm(sz, device=device)
        # return torch.arange(sz).to(device)

    def get_dummy_input(self, bs=1):
        ret = {}
        device = self.global_configs["training"]["device"]
        ret['sdf'] = torch.zeros(bs, 1, 64, 64, 64).to(device)
        ret['idx'] = torch.zeros(
            bs, self.grid_size, self.grid_size, self.grid_size).long().to(device)
        ret['z_q'] = torch.zeros(
            bs, 256, self.grid_size, self.grid_size, self.grid_size).to(device)

        return ret

    def set_input(self, input=None, gen_order=None):

        self.x = input['sdf']
        self.x_idx = input['idx']
        self.z_q = input['z_q']
        bs, dz, hz, wz = self.x_idx.shape
        self.z_shape = self.z_q.shape

        self.x_idx_seq = rearrange(
            self.x_idx, 'bs dz hz wz -> (dz hz wz) bs').contiguous()  # to (T, B)
        self.x_idx = self.x_idx_seq.clone()

        # prepare input for transformer
        T, B = self.x_idx.shape[:2]

        if gen_order is None:
            self.gen_order = self.get_gen_order(T, self.x.device)
            self.context_len = -1  # will be specified in inference
        else:
            if len(gen_order) != T:

                self.context_len = len(gen_order)
                # pad the remaining
                remain = torch.tensor(
                    [i for i in range(T) if i not in gen_order]).to(gen_order)
                remain = remain[torch.randperm(len(remain))]
                self.gen_order = torch.cat([gen_order, remain])
            else:
                self.gen_order = gen_order

        x_idx_seq_shuf = self.x_idx_seq[self.gen_order]
        x_seq_shuffled = torch.cat(
            [torch.LongTensor(1, bs).fill_(self.sos).to(self.x.device), x_idx_seq_shuf], dim=0).to(self.x.device)  # T+1
        # T+1, <sos> should always at start.
        pos_shuffled = torch.cat(
            [self.grid_table[:1], self.grid_table[1:][self.gen_order.cpu()]], dim=0).to(self.x.device)

        self.inp = x_seq_shuffled[:-1].clone()
        self.tgt = x_seq_shuffled[1:].clone()
        self.inp_pos = pos_shuffled[:-1].clone()
        self.tgt_pos = pos_shuffled[1:].clone()

    def forward(self, data):
        """ given 
                inp, inp_pos, tgt_pos
            infer
                tgt
            outp is the prob. dist. over x_(t+1) at pos_(t+1)
            p(x_{t+1} | x_t, pos_t, pos_{t+1})
        """
        self.tf.train()
        self.set_input(data)
        self.outp = self.tf(self.inp, self.inp_pos, self.tgt_pos)  # [:-1]
        return self.outp

    def inference2(self, data, seq_len=None, gen_order=None, topk=None, prob=None, alpha=1., should_render=False, verbose=False):
        def top_k_logits(logits, k=5):
            v, ix = torch.topk(logits, k)
            out = logits.clone()
            out[out < v[:, :, [-1]]] = -float('Inf')
            return out

        self.tf.eval()

        # context:
        #     - if prob is given, seq_len=1
        #     - else seq_len is defined by gen_order
        if prob is not None:
            if seq_len is None:
                seq_len = 1  # context
        else:
            if gen_order is None:
                if seq_len is None:
                    seq_len = 1  # context
            else:
                # if goes here, context_len will be given by gen_order
                # +1 to include sos
                seq_len = len(gen_order)+1

        self.set_input(data, gen_order=gen_order)

        T = self.x_idx_seq.shape[0] + 1  # +1 since <sos>
        B = self.x_idx_seq.shape[1]

        if prob is not None:
            prob = prob[self.gen_order]
            prob = torch.cat([prob[:1], prob])

        with torch.no_grad():
            # auto-regressively gen
            pred = self.inp[:seq_len]
            for t in tqdm(range(seq_len, T), total=T-seq_len, desc='[*] autoregressively inferencing...'):
                inp = pred
                inp_pos = self.inp_pos[:t]
                tgt_pos = self.tgt_pos[:t]
                # inp_mask = self.generate_square_subsequent_mask(
                #     transformer_inp.shape[0], self.opt.device)
                outp = self.tf(inp, inp_pos, tgt_pos)
                outp_t = outp[-1:]
                # outp_t = F.softmax(outp_t, dim=-1)  # compute prob
                outp_t = F.log_softmax(outp_t, dim=-1)

                if prob is not None:
                    # outp_t *= prob[t:t+1]
                    # outp_t += prob[t:t+1] # logspace
                    outp_t = (1-alpha) * outp_t + alpha * prob[t:t+1]

                if topk is not None:
                    # outp_t = top_k_probs(outp_t, k=topk)
                    outp_t = top_k_logits(outp_t, k=topk)

                outp_t = F.softmax(outp_t, dim=-1)  # compute prob
                outp_t = rearrange(outp_t, 't b nc -> (t b) nc')
                pred_t = torch.multinomial(outp_t, num_samples=1).squeeze(1)
                pred_t = rearrange(pred_t, '(t b) -> t b', t=1, b=B)
                pred = torch.cat([pred, pred_t], dim=0)

            self.x = self.x
            self.x_recon = self.vqvae.decode(
                self.z_q)  # could extract this as well
            # exclude pred[0] since it's <sos>
            pred = pred[1:][torch.argsort(self.gen_order)]
            self.x_recon_tf = self.vqvae.decode_enc_idices(
                pred, z_spatial_dim=self.grid_size)
            self.tf.train()
            return self.x_recon_tf

    def inference(self, data):
        self.tf.eval()
        self.set_input(data)
        self.outp = self.tf(self.inp, self.inp_pos, self.tgt_pos)

    def set_loss(self):
        target = rearrange(self.tgt, 'seq b -> (seq b)')
        outp = rearrange(self.outp, 'seq b cls-> (seq b) cls')
        loss_nll = self.criterion_ce(outp, target)
        self.loss = loss_nll

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def step(self, x):
        self.train()
        self.set_requires_grad([self.tf], requires_grad=True)
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def get_metrics(self, apply_additional_metrics=False):
        return {'loss': self.loss.data}

    def set_iteration(self, iteration):
        self.iteration = iteration

    def prepare_visuals(self):
        with torch.no_grad():
            outp = F.softmax(self.outp, dim=-1)
            outp = outp.argmax(dim=-1)
            outp = outp[torch.argsort(self.gen_order)]
            self.x_recon_tf = self.vqvae.decode_enc_idices(outp)
            self.x_recon = self.vqvae.decode(
                self.z_q)

        visuals = {
            "reconstructions_tf": self.x_recon_tf,
            "reconstructions_pvqvae": self.x_recon,
            "target": self.x,
        }
        uncond = self.uncond_gen(2)
        visuals["generated"] = uncond
        return visuals

    def get_batch_input(self, x):
        return x

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def uncond_gen(self, bs=1, topk=30):

        # get dummy data
        data = self.get_dummy_input(bs=bs)
        self.inference2(data, seq_len=None, topk=topk)
        gen_tf = self.x_recon_tf
        return gen_tf

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))
        cprint.info(f"Model loaded from {ckpt_path}")
        state_dict = torch.load(self.configs['pvqvae']['ckpt_path'])
        self.vqvae.load_state_dict(state_dict)
        self.tf.embedding_encoder.load_state_dict(
            self.vqvae.quantize.embedding.state_dict())

        cprint.info(f"VQVAE loaded from {self.configs['pvqvae']['ckpt_path']}")
