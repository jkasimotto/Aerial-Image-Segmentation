import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class NoOP(nn.Module):
    """
    This class is a Module that performs no operation and is used when the attention layer of the UnetBlock is not specified.
    """

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
        self.conv = nn.Sequential(
            # Set bias to False because we use BatchNorm.
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),

            # Set in_channels=outchannels to compose with the first conv.
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
        # Return all features for skip connections.
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.encoder4(self.pool(x3))
        x5 = self.encoder5(self.pool(x4))
        return x1, x2, x3, x4, x5


class UnetDecoder(nn.Module):

    def __init__(self, features, out_channels, attn) -> None:
        super().__init__()
        # Reverse features to decode in reverse
        features = features[::-1]

        # The 4 transpose convolutions used in the decoder.
        self.up4 = nn.ConvTranspose2d(
            in_channels=features[0]*2, out_channels=features[0], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(
            in_channels=features[1]*2, out_channels=features[1], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(
            in_channels=features[2]*2, out_channels=features[2], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(
            in_channels=features[3]*2, out_channels=features[3], kernel_size=2, stride=2)
        # The 4 UnetBlocks used in the decoder.
        self.decoder4 = UnetBlock(
            in_channels=features[0]*2, out_channels=features[0], attn=attn)
        self.decoder3 = UnetBlock(
            in_channels=features[1]*2, out_channels=features[1], attn=attn)
        self.decoder2 = UnetBlock(
            in_channels=features[2]*2, out_channels=features[2], attn=attn)
        self.decoder1 = UnetBlock(
            in_channels=features[3]*2, out_channels=features[3], attn=attn)
        # The final convolutional layer that outputs a single channel.
        self.final = nn.Conv2d(
            in_channels=features[3], out_channels=out_channels, kernel_size=1)

    def forward(self, features):
        # Pass in the features from the encoder to concatenate as skip connections.
        e1, e2, e3, e4, e5 = features

        d5 = self.up4(e5)

        if d5.shape != e4.shape:
            d5 = TF.resize(d5, size=e4.shape[2:])
        d4 = torch.cat([e4, d5], dim=1)
        d4 = self.decoder4(d4)
        d4 = self.up3(d4)

        if d4.shape != e3.shape:
            d4 = TF.resize(d4, size=e3.shape[2:])
        d3 = torch.cat([e3, d4], dim=1)
        d3 = self.decoder3(d3)
        d3 = self.up2(d3)

        if d3.shape != e2.shape:
            d3 = TF.resize(d3, size=e2.shape[2:])
        d2 = torch.cat([e2, d3], dim=1)
        d2 = self.decoder2(d2)
        d2 = self.up1(d2)

        if d2.shape != e1.shape:
            d2 = TF.resize(d2, size=e1.shape[2:])
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
