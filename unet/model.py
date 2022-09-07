import torch
import torch.nn as nn
from fastai.vision.all import noop


class UnetBlock(nn.Module):
    """
    This class implements the double convolution used in both the encoder and decoder of the UNET model.
    """

    def __init__(self, in_channels, out_channels, attn=None) -> None:
        super().__init__()
        # Set bias to False because we use BatchNorm.
        # Set in_channels=outchannels in the second convolution to enable composition.
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1 bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1 bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.attn = attn(out_channels) if attn else noop

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.attn(x)
        return x


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
        x1 = self.pool(x1)

        x2 = self.encoder2(x1)
        x2 = self.pool(x2)

        x3 = self.encoder3(x2)
        x3 = self.pool(x3)

        x4 = self.encoder4(x3)
        x4 = self.pool(x4)

        x5 = self.encoder5(x4)
        return x1, x2, x3, x4, x5



class UnetDecoder(nn.Module):

    def __init__(self, features, out_channels, attn) -> None:
        super().__init__()
        # What goes down must come up... in reverse
        features = reversed(features)

        self.up4 = nn.ConvTranspose2d(in_channels=features[0]*2, out_channels=features[0])
        self.up3 = nn.ConvTranspose2d(in_channels=features[1]*2, out_channels=features[1])
        self.up2 = nn.ConvTranspose2d(in_channels=features[2]*2, out_channels=features[2])
        self.up1 = nn.ConvTranspose2d(in_channels=features[3]*2, out_channels=features[3])
        self.decoder4 = UnetBlock(in_channels=features[0]*2, out_channels=features[0], attn=attn)
        self.decoder3 = UnetBlock(in_channels=features[1]*2, out_channels=features[1], attn=attn)
        self.decoder2 = UnetBlock(in_channels=features[2]*2, out_channels=features[2], attn=attn)
        self.decoder1 = UnetBlock(in_channels=features[3]*2, out_channels=features[3], attn=attn)
        self.final = nn.Conv2d(in_channels=features[3], out_channels=out_channels, kernel_size=1)

    def forward(self, features):
        e1, e2, e3, e4, e5 = features

        d5 = self.up4(e5)

        d4 = self.decoder4(torch.cat([e4, d5]))
        d4 = self.up3(d4)

        d3 = self.decoder3(torch.cat([e3, d4]))
        d3 = self.up2(d3)

        d2 = self.decoder2(torch.cat([e2, d3]))
        d2 = self.up1(d2)

        d1 = self.decoder1(torch.cat([e1, d2]))
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
<<<<<<< HEAD
        features = self.encoder(x)
        out = self.decoder(features)
        return out
=======
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # We append downsampled layers. The first is the one with the highest resolution (no downsampling), the last will be the lowest resolution (all downsampled)
            x = self.pool(x)
        
        # After downsampling go through the bottleneck layer.
        x = self.bottleneck(x)
        
        # When going through the up layers we concat skip_connections in reverse order so I reverse them here.
        skip_connections = skip_connections[::-1] 

        # Step size of 2 because when we created self.ups the 2 convolutions we appended are considered as a single step.
        for idx in range(0, len(self.ups), 2):
            # Upsample with ConvTranspose2d 
            x = self.ups[idx](x) 

            # Concatenate the corresponding skip_connection.
            # If the input dimenions are not perfectly divisible by (2**#down_samples) then downsampling will reduce the size of x and concat won't work. 
            # We solve this by resizing x.
            skip_connection = skip_connections[idx//2] # Floor division gets the skip_connections linearly.
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:]) # Only resize along height and width leaving batch_size and channels untouched.
            concat_skip = torch.cat((skip_connection, x), dim=1) # Concat along channel dimension.

            # The DoubleConv
            x = self.ups[idx+1](concat_skip)
        
        return self.final_conv(x)
>>>>>>> unet-train-v2
