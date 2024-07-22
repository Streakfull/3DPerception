from src.models.base_model import BaseModel
from src.blocks.encoder import Encoder
from src.blocks.decoder import Decoder
from torch import optim
from src.losses.build_loss import BuildLoss
from src.utils.model_utils import init_weights
from torch import nn


class AutoEncoder(BaseModel):
    def __init__(self, configs):
        super().__init__()
        self.encoder = Encoder(**(configs["auto_encoder_networks"]))
        decoder_config = configs["auto_encoder_networks"]
        decoder_config["in_channels"] = self.encoder.out_channels
        self.decoder = Decoder(**decoder_config)
        self.criterion = BuildLoss(configs).get_loss()

        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        self.configs = configs
        init_type = self.configs['weight_init']
        if (init_type != "None"):
            print("Initializing model weights with %s initialization" % init_type)
            self.init_weights()
        self.set_metrics()

    def forward(self, x):
        self.target = x
        x = self.encoder(x)
        x = self.decoder(x)
        self.predictions = x
        return x

    def set_loss(self):
        self.loss = self.criterion(self.predictions, self.target)

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
        if (not apply_additional_metrics):
            return {'loss': self.loss.data}
        metrics = {'loss': self.loss.data}
        for metric in self.metrics:
            value = metric[1].calc_batch(self.predictions, self.target)
            metrics[metric[0]] = value
        return metrics

    def inference(self, x):
        self.eval()
        x = self.forward(x)
        return x

    def init_weights(self):
        init_type = self.configs['weight_init']
        gain = self.configs['gain']
        init_weights(self.encoder, init_type=init_type, gain=gain)
        init_weights(self.decoder, init_type=init_type, gain=gain)

    def get_batch_input(self, x):
        return x['sdf']

    def prepare_visuals(self):
        visuals = {
            "reconstructions": self.predictions,
            "target": self.target,


        }
        return visuals

    def calculate_additional_metrics(self):
        metrics = {}
        for metric in self.metrics:
            value = metric[1].calc_batch(self.predictions, self.target)
            metrics[metric[0]] = value
        return metrics
