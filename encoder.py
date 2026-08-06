import torch
from torch import nn


class Encoder3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.globalPool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # print("After block 1:", x.shape)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print("After block 2:", x.shape)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.globalPool(x)
        x = torch.flatten(x, start_dim=1)

        return x

if __name__ == "__main__":
    model = Encoder3D()

    fakeVolume = torch.randn(2, 1, 32, 32, 32)
    output = model(fakeVolume)

    print("Input shape:       ", fakeVolume.shape)
    print("Final output shape:", output.shape)
