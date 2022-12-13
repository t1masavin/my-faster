import torch.nn as nn


class ConvBatchNormReLU(nn.Module):
    def __init__(self, input_channels=64, output_channels=64, kernel_size=(3, 3), stride=1, pad=1) -> None:
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride, pad)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DownSample(nn.Module):
    def __init__(self, kernel_size=2, stride=2) -> None:
        super().__init__()
        self.pooling = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pooling(x)


class VGG_conv(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.conv1 = self.build_block(in_channels, 64)
        self.down1 = DownSample(2, 2)
        self.conv2 = self.build_block(64, 128, num_layers=3)
        self.down2 = DownSample(2, 2)
        self.conv3 = self.build_block(128, 256, num_layers=3)

    def build_block(self, input_channels=64, output_channels=64, kernel_size=(3, 3), stride=1, pad=1, num_layers=2):
        layers = [ConvBatchNormReLU(input_channels, output_channels, kernel_size, stride, pad)]
        for _ in range(1, num_layers):
            layers.append(ConvBatchNormReLU(output_channels, output_channels, kernel_size, stride, pad))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.down1(x)
        x = self.conv2(x)
        x = self.down2(x)
        x = self.conv3(x)
        return x
