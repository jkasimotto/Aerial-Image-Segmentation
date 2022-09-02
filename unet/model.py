import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    # A class representing the double convolution used in the UNet model. 
    # There are two iterations of a same convolution followed by batchnorm and ReLU activation.
    # Copied from this video https://www.youtube.com/watch?v=IHq1t7NxS8k
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # Same convolution. 
            # Stide and padding = 1 means input height and width is the same after the convolution.
            # Bias is set to false to use BatchNorm afterwards.
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), 

            # Repeat the steps but set in_channels=out_channels to allow for the composition.
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.conv(x)





class UNET(nn.Module):
    def __init__(
        self,  
        in_channels=3, # Takes in RGB channels.
        out_channels=1, # Outputs a binary channel.
        features=[64,128,256,512] # The size of the feature maps in the downsampling & upsampling.
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList() # ModuleList is required for using model.eval() model.train() etc
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        # Create the down part of UNET.
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature # Set in_channels equal to the output_channels of the previous layer for the next down layer.
        
        # Create the up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, # Set in_channels to 2*out_channels of last down layer because the skip connection concatenates a feature map.
                    feature,
                    kernel_size=2,
                    stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))
        
        # This layer is the DoubleConv that exists outside of downsampling/upsampling and occurs after the downsampling.
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # This is the final conv layer to change the number of channels.
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        print("Going down")
        for down in self.downs:
            x = down(x)
            skip_connections.append(x) # We append downsampled layers. The first is the one with the highest resolution (no downsampling), the last will be the lowest resolution (all downsampled)
            x = self.pool(x)
        
        print("Bottleneck")
        # After downsampling go through the bottleneck layer.
        x = self.bottleneck(x)
        
        print("Going up")
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