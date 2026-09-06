import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Structure: 4 conv -> relu -> maxpool blocks, doubling the number of
    channels at each block (1 -> 16 -> 32 -> 64 -> 128) and halving the
    spatial size each time (224 -> 112 -> 56 -> 28 -> 14), followed by
    global average pooling (collapses each feature map to 1x1, regardless
    of the input spatial size) and a single linear layer producing the
    2 output logits (one per class, to be fed into CrossEntropyLoss).
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,16,3,padding=1)
        self.maxPool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.maxPool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        self.maxPool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        self.maxPool4 = nn.MaxPool2d(2)
        self.globalmean = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Linear(128,2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxPool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxPool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxPool3(x)
        x = F.relu(self.conv4(x))
        x = self.maxPool4(x)
        x = self.globalmean(x)
        x = torch.flatten(x,1)
        x = self.linear1(x)
        return x

