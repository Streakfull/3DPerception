import os
from termcolor import cprint
import torch


# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix


class BaseModel(torch.nn.Module):
    @staticmethod
    def name():
        return 'BaseModel'

    def __init__(self):
        super().__init__()

    def initialize(self, opt):
        self.opt = opt
        self.is_train = opt.is_train

        if self.is_train:
            self.save_dir = os.path.join(opt.logs_dir, opt.name, 'checkpoints')

            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
                cprint(f"{self.save_dir} created", "blue")

        self.model_names = []
        self.epoch_labels = []
        self.optimizer = None
        self.scheduler = None

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def backward(self):
        pass

    def step(self):
        pass

    def save(self, path, epoch="latest"):
        checkpoint_name = f"{path}/epoch-{epoch}.ckpt"
        cprint(f"{checkpoint_name} created", "blue")
        torch.save(self.state_dict(), checkpoint_name)

    def inference(self):
        pass

    def get_loss(self):
        return 0

    def update_lr(self):
        if (self.scheduler is None):
            return
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        cprint('[*] learning rate = %.7f' % lr, "yellow")

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))
        cprint(f"Model loaded from {ckpt_path}")

    def tocuda(self, var_names):
        for name in var_names:
            if isinstance(name, str):
                var = getattr(self, name)
                setattr(self, name, var.cuda(0, non_blocking=True))

    def init_weights(self):
        pass

    def get_batch_input(self, x):
        return x
