from torch import nn
from src.losses.dice_loss import DiceLoss
from src.losses.KL_divergence import KLDivergence
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

            case "MSE":
                return nn.MSELoss()

            case "KL":
                return KLDivergence()

        return nn.CrossEntropyLoss()
