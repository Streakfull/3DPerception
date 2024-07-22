from src.models.base_model import BaseModel
from torch import nn
from torch import optim
import torch
import json
import os
from pathlib import Path


def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = nn.Sequential(
        nn.Conv3d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        nn.BatchNorm3d(chann_out),
        nn.ReLU()
    )
    return layer


def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):

    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i])
              for i in range(len(in_list))]
    layers += [nn.MaxPool3d(kernel_size=pooling_k, stride=pooling_s)]
    return nn.Sequential(*layers)


def vgg_fc_layer(size_in, size_out):
    layer = nn.Sequential(
        nn.Linear(size_in, size_out),
        nn.BatchNorm1d(size_out),
        nn.ReLU()
    )
    return layer


class VGG16(BaseModel):
    def __init__(self, configs):
        super(VGG16, self).__init__()
        self.class_name_mapping = json.loads(
            Path("src/datasets/shape_net/shape_info.json").read_text())
        self.classes = sorted(self.class_name_mapping.keys())
        # Conv blocks (BatchNorm + ReLU activation added in each block)
        self.layer1 = vgg_conv_block([1, 64], [64, 64], [3, 3], [1, 1], 2, 2)
        self.layer2 = vgg_conv_block(
            [64, 128], [128, 128], [3, 3], [1, 1], 2, 2)
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [
                                     3, 3, 3], [1, 1, 1], 2, 2)

        # FC layers
        self.layer6 = vgg_fc_layer(4096, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)
        self.configs = configs
        # Final layer
        self.layer8 = nn.Linear(4096, 13)
        self.optimizer = optim.Adam(params=self.parameters(), lr=configs["lr"])
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=configs["scheduler_step_size"], gamma=configs["scheduler_gamma"])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        self.input = x
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        self.out = self.layer8(out)
        return self.out, vgg16_features

    def backward(self):
        self.set_loss()
        self.loss.backward()

    def set_loss(self):
        self.target = self.target.to(self.out.device)
        self.loss = self.criterion(self.out, self.target)
        return self.loss

    def step(self, x):
        self.train()
        self.optimizer.zero_grad()
        x = self.forward(x)
        self.backward()
        self.optimizer.step()

    def get_batch_input(self, x):
        self.target = x['label']

        return x['sdf']

    def inference(self, x):
        self.eval()
        x = self.forward(x)
        return x

    def get_metrics(self):
        return {'loss': self.loss.data, 'acc': self.accuracy()}

    def prepare_visuals(self):
        visuals = {
            "target": self.input,
        }
        return visuals

    def prepare_text_visuals(self):
        predictions = nn.functional.softmax(self.out.detach())
        predictions = torch.argmax(predictions, dim=1)
        text = []
        for pred in predictions:
            class_name = self.class_name_mapping[self.classes[pred]]
            text.append(class_name)
        return " ,".join(text)

    def accuracy(self):
        predictions = nn.functional.softmax(self.out.detach())
        predictions = torch.argmax(predictions, dim=1)
        total_correct = predictions == self.target
        acc = total_correct.sum()/predictions.shape[0]
        return acc
