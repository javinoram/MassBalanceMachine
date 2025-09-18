import sys, os
sys.path.append(os.path.join(os.getcwd(), '../')) # Add root of repo to import MBM

import pandas as pd
import massbalancemachine as mbm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
from skorch.helper import SliceDataset
import torchsummary

warnings.filterwarnings('ignore')

class SimpleNetwork(nn.Module):
    def __init__(self, nNeurons, *args, **kwargs):
        assert len(nNeurons)>=2
        super().__init__(*args, **kwargs)
        layers = nn.Sequential(nn.Linear(nNeurons[0], nNeurons[1]))
        for i in range(1, len(nNeurons)-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(nNeurons[i], nNeurons[i+1]))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(nNeurons[-1], 1))
        self.layers = layers
    def forward(self, x):
        return self.layers(x)


