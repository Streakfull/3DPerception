#adapted from https://github.com/fomalhautb/3D-RETR
from torch import nn
import torch
from losses.dice_loss import DiceLoss

class DiceLossBCE(nn.Module):
    def __init__(self, smooth=1, reduction='mean' , pos_weight = None):
        super(DiceLossBCE, self).__init__()
        self.smooth = smooth
        self.dice = DiceLoss()
        if(pos_weight != None):
            self.cross_entropy = nn.BCEWithLogitsLoss(reduction=reduction, pos_weight=torch.tensor(pos_weight))
        else:
            self.cross_entropy = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, output, target):
        bce = self.cross_entropy(output, target)
        dice = self.dice(output, target)
        return bce + dice