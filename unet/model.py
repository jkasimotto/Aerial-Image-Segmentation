import torch
import torch.nn as nn
from fastai.vision.all import noop


class NoOP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


class UnetBlock(nn.Module):
    """
    This class implements the double convolution used in both the encoder and decoder of the UNET model.
    """

    def __init__(self, in_channels, out_channels, attn=None) -> None:
        super().__init__()
        # Set bias to False because we use BatchNorm.
        # Set in_channels=outchannels in the second convolution to enable composition.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            attn(out_channels, inplace=True) if attn else NoOP()
        )

    def forward(self, x):
        return self.conv(x)


class UnetEncoder(nn.Module):

    def __init__(self, in_channels, features) -> None:
        super().__init__()
        self.encoder1 = UnetBlock(in_channels, features[0])
        self.encoder2 = UnetBlock(features[0], features[1])
        self.encoder3 = UnetBlock(features[1], features[2])
        self.encoder4 = UnetBlock(features[2], features[3])
        self.encoder5 = UnetBlock(
            features[3], features[3] * 2)  # The centre/bottleneck layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Maintain the features for skip connections.
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.encoder4(self.pool(x3))
        x5 = self.encoder5(self.pool(x4))
        return x1, x2, x3, x4, x5


class UnetDecoder(nn.Module):

    def __init__(self, features, out_channels, attn) -> None:
        super().__init__()
        # What goes down must come up... in reverse
        features = features[::-1]

        self.up4 = nn.ConvTranspose2d(
            in_channels=features[0]*2, out_channels=features[0], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(
            in_channels=features[1]*2, out_channels=features[1], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(
            in_channels=features[2]*2, out_channels=features[2], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(
            in_channels=features[3]*2, out_channels=features[3], kernel_size=2, stride=2)
        self.decoder4 = UnetBlock(
            in_channels=features[0]*2, out_channels=features[0], attn=attn)
        self.decoder3 = UnetBlock(
            in_channels=features[1]*2, out_channels=features[1], attn=attn)
        self.decoder2 = UnetBlock(
            in_channels=features[2]*2, out_channels=features[2], attn=attn)
        self.decoder1 = UnetBlock(
            in_channels=features[3]*2, out_channels=features[3], attn=attn)
        self.final = nn.Conv2d(
            in_channels=features[3], out_channels=out_channels, kernel_size=1)

    def forward(self, features):
        e1, e2, e3, e4, e5 = features

        d5 = self.up4(e5)

        d4 = torch.cat([e4, d5], dim=1)
        d4 = self.decoder4(d4)
        d4 = self.up3(d4)

        d3 = torch.cat([e3, d4], dim=1)
        d3 = self.decoder3(d3)
        d3 = self.up2(d3)

        d2 = torch.cat([e2, d3], dim=1)
        d2 = self.decoder2(d2)
        d2 = self.up1(d2)

        d1 = torch.cat([e1, d2], dim=1)
        d1 = self.decoder1(d1)
        d1 = self.final(d1)

        return d1


class UNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=1,
        features_sizes=[64, 128, 256, 512],
        attn=None
    ) -> None:
        super().__init__()
        self.encoder = UnetEncoder(in_channels, features_sizes)
        self.decoder = UnetDecoder(features_sizes, out_channels, attn)

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
