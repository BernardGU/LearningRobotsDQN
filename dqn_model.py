import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class DQN_Model(nn.Module):
    """ This model follows the architecture proposed in https://daiwk.github.io/assets/dqn.pdf
    """
    def __init__(self, in_channels:int=4, dims:Tuple[int, int]=(84,84), outputs:int=18):
        super(DQN_Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False)
        self.fc1   = nn.Linear(64*7*7, 512)
        self.out   = nn.Linear(512, outputs)

        self.in_channels = in_channels
        self.dims        = dims
        self.outputs     = outputs
        
    def forward(self, x):
        # Hidden Convolutional Layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Hidden Linear Layer
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        # Output Layer
        x = self.out(x)

        return x