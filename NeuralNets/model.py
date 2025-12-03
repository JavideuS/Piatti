import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)  # Add skip connection
        out = self.relu(out)

        return out


class PiattiCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Residual blocks
            ResidualBlock(64, 64),
            nn.MaxPool2d(2),

            ResidualBlock(64, 128, stride=1),
            nn.MaxPool2d(2),

            ResidualBlock(128, 256, stride=1),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
            # nn.Linear(256, 128),
            # nn.ReLU(inplace=True),
            # nn.Dropout(0.3),
            # nn.Linear(128, num_classes)

            # Note that would simply add more parameters but we don't have enough classes or images for it to be useful
            # In fact it performs worse with these extra layers
            # nn.Linear(256, 128),
            # nn.Linear(128, 64),
            # nn.Linear(64, num_classes)

            #Should attach softmax at the end to get a probability distribution???
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

