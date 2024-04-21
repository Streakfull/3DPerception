from torch import nn
from src.losses.dice_loss import DiceLoss
import torch


class BuildLoss:
    def __init__(self, configs):
        self.configs = configs

    def get_loss(self):
        match self.configs["criterion"]:
            case "BCE":
                if (self.pos_weight == None):
                    return nn.BCEWithLogitsLoss()
                else:
                    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.pos_weight))
            case "DICE":
                return DiceLoss()

            case "CE":
                return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss()
