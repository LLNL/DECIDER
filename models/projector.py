import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
import clip

from functools import reduce
from operator import mul
import numpy as np
from PIL import Image
import math

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim,is_mlp=False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        if is_mlp:
            self.linear = nn.Sequential(
                nn.Linear(input_dim, 1024),
                nn.ReLU(),
                nn.Linear(1024, output_dim),
            )
        else:
            self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        return x

class ProjectionCNN(nn.Module):
    def __init__(self, input_channels, input_height, input_width, embedding_dim):
        super(ProjectionCNN, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(input_channels, input_channels/2, kernel_size=3, stride=1, padding=1),  
            nn.Flatten(),
            nn.Linear(input_channels/2 * input_height * input_width, embedding_dim)
        )

    def forward(self, x):
        return self.proj(x) 
