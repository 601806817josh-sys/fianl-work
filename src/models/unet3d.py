import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DownBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class UpBlock3D(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)

        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)

        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=(32, 64, 128, 256)):
        super().__init__()

        self.stem = DoubleConv3D(in_channels, features[0])

        self.down1 = DownBlock3D(features[0], features[1])
        self.down2 = DownBlock3D(features[1], features[2])
        self.down3 = DownBlock3D(features[2], features[3])

        self.bottleneck = DownBlock3D(features[3], features[3] * 2)

        self.up3 = UpBlock3D(features[3] * 2, features[3], features[3])
        self.up2 = UpBlock3D(features[3], features[2], features[2])
        self.up1 = UpBlock3D(features[2], features[1], features[1])
        self.up0 = UpBlock3D(features[1], features[0], features[0])

        self.head = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)

        xb = self.bottleneck(x3)

        x = self.up3(xb, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        x = self.up0(x, x0)

        logits = self.head(x)
        return logits


if __name__ == "__main__":
    model = UNet3D()
    input_tensor = torch.randn(1, 1, 64, 64, 64)
    output_tensor = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
