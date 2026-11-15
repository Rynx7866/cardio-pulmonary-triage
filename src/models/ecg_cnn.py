"""1D CNN with residual blocks for PTB-XL multilabel classification."""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """A 1D residual convolution block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=stride,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        return self.relu(out)


class ECGCNN(nn.Module):
    """
    Residual 1D CNN for five-class multilabel ECG classification.

    Input:  (batch_size, 12, time_steps)
    Output: (batch_size, 5) raw logits
    """

    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                12, 32, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualBlock(32, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (batch, 12, time), got {tuple(x.shape)}"
            )

        if x.shape[1] != 12:
            raise ValueError(
                f"Expected 12 ECG leads, got {x.shape[1]}"
            )

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.pool(x)
        return self.classifier(x)
