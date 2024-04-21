import torch.nn as nn
from einops import rearrange


# A dummy network composed of a deep CNN for classification


# (1, 32, 32, 32)
class DummyNetwork(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=4,
                      kernel_size=4, stride=3, padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Conv3d(in_channels=4, out_channels=8,
                      kernel_size=4, stride=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16,
                      kernel_size=4, stride=3, padding=1),
            nn.ReLU(),

        )
        self.mlp = nn.Sequential(
            nn.Linear(in_features=16, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=13),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn(x)
        x = rearrange(x, 'bs ch l w h -> bs (ch l w h)')
        x = self.mlp(x)
        return x
