import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time

class ResNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, hidden3_dim, output_dim):
        super(ResNetClassifier, self).__init__()

        # Store input_dim as an attribute to be compatible with PytorchModel
        self.input_dim = input_dim

        # First layer
        self.hidden1 = nn.Linear(input_dim, hidden1_dim)
        self.bn1 = nn.BatchNorm1d(hidden1_dim)
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

        # Second layer with residual connection
        self.hidden2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.bn2 = nn.BatchNorm1d(hidden2_dim)
        self.act2 = nn.ReLU()
        # Identity mapping for residual connection (if dimensions differ)
        self.identity1 = nn.Identity() if hidden1_dim == hidden2_dim else nn.Linear(hidden1_dim, hidden2_dim)

        # Third layer with residual connection
        self.hidden3 = nn.Linear(hidden2_dim, hidden3_dim)
        self.bn3 = nn.BatchNorm1d(hidden3_dim)
        self.act3 = nn.ReLU()
        # Identity mapping for residual connection (if dimensions differ)
        self.identity2 = nn.Identity() if hidden2_dim == hidden3_dim else nn.Linear(hidden2_dim, hidden3_dim)

        # Output layer
        self.output = nn.Linear(hidden3_dim, output_dim)

    def forward(self, x):
        # First hidden layer
        if x.size(0) > 1:
            h1 = self.act1(self.bn1(self.hidden1(x)))
        else:
            h1 = self.act1(self.hidden1(x))
        h1 = self.dropout(h1)

        # Second hidden layer with residual connection
        identity1 = self.identity1(h1)
        if x.size(0) > 1:
            h2_linear = self.hidden2(h1)
            h2_bn = self.bn2(h2_linear)
            h2_act = self.act2(h2_bn)
        else:
            h2_linear = self.hidden2(h1)
            h2_act = self.act2(h2_linear)

        h2 = h2_act + identity1  # Add residual connection
        h2 = self.dropout(h2)

        # Third hidden layer with residual connection
        identity2 = self.identity2(h2)
        if x.size(0) > 1:
            h3_linear = self.hidden3(h2)
            h3_bn = self.bn3(h3_linear)
            h3_act = self.act3(h3_bn)
        else:
            h3_linear = self.hidden3(h2)
            h3_act = self.act3(h3_linear)

        h3 = h3_act + identity2  # Add residual connection
        h3 = self.dropout(h3)

        # Output layer
        logits = self.output(h3)
        return logits

    def extract_features(self, x):
        # First hidden layer
        if x.size(0) > 1:
            h1 = self.act1(self.bn1(self.hidden1(x)))
        else:
            h1 = self.act1(self.hidden1(x))

        # Second hidden layer with residual connection
        identity1 = self.identity1(h1)
        if x.size(0) > 1:
            h2_linear = self.hidden2(h1)
            h2_bn = self.bn2(h2_linear)
            h2_act = self.act2(h2_bn)
        else:
            h2_linear = self.hidden2(h1)
            h2_act = self.act2(h2_linear)

        h2 = h2_act + identity1  # Add residual connection

        # Third hidden layer with residual connection
        identity2 = self.identity2(h2)
        if x.size(0) > 1:
            h3_linear = self.hidden3(h2)
            h3_bn = self.bn3(h3_linear)
            h3_act = self.act3(h3_bn)
        else:
            h3_linear = self.hidden3(h2)
            h3_act = self.act3(h3_linear)

        h3 = h3_act + identity2  # Add residual connection

        return h3
    

