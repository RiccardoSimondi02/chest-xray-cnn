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


class ModelLargeHead(nn.Module):
    """
    Variant of Model without global average pooling: the 4 conv -> relu ->
    maxpool blocks are the same (channels 1 -> 16 -> 32 -> 64 -> 128,
    spatial size 224 -> 112 -> 56 -> 28 -> 14), but the resulting
    128x14x14 feature map is flattened as-is (preserving spatial
    information instead of averaging it away) and fed into a 2-layer MLP
    head (Linear(128*14*14, 128) -> relu -> Linear(128, 2)).

    Note: this was tested to check whether dropping GAP helps. It
    overfits (high train balanced accuracy) more than the GAP version.
    """
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.maxPool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.maxPool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.maxPool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.maxPool4 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(128 * 14 * 14, 128)
        self.linear2 = nn.Linear(128, 2)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = self.maxPool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxPool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxPool3(x)
        x = F.relu(self.conv4(x))
        x = self.maxPool4(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

