# Replace UNet's blocks with SplConv2d for extracting multi-scale features
import torch
import torch.nn as nn
import math
import torch.nn.functional as F 
from torch.nn import Conv2d, Linear, BatchNorm2d, ReLU

class DynamicSplConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, num_splits):
        super().__init__()
        self.num_splits = num_splits

        # Define convolution layers with varying kernel sizes
        kernel_sizes = [1 + 2 * i for i in range(num_splits)]  # Example: [1, 3, 5, 7] for 4 splits
        self.convs = nn.ModuleList()

        # Calculate base split size and remainder for input and output channels
        base_in_channels = in_channels // num_splits
        remainder_in_channels = in_channels % num_splits
        base_out_channels = out_channels // num_splits
        remainder_out_channels = out_channels % num_splits

        # Create convolution layers dynamically based on the number of splits
        for i in range(num_splits):
            in_channels_split = base_in_channels + (1 if i < remainder_in_channels else 0)
            out_channels_split = base_out_channels + (1 if i < remainder_out_channels else 0)

            self.convs.append(nn.Sequential(
                nn.Conv2d(in_channels_split, out_channels_split, kernel_size=kernel_sizes[i], padding=kernel_sizes[i] // 2),
                nn.BatchNorm2d(out_channels_split),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        batch, channel = x.shape[:2]

        # Calculate base split size and remainder for input channels
        base_split_size = channel // self.num_splits
        remainder = channel % self.num_splits

        # Split the input into num_splits parts, assigning the remainder to the first few splits
        split_sizes = [(base_split_size + 1 if i < remainder else base_split_size) for i in range(self.num_splits)]
        split_tensors = torch.split(x, split_sizes, dim=1)

        # Apply convolution to each split
        conv_results = [conv(split) for conv, split in zip(self.convs, split_tensors)]

        # Concatenate the results along the channel dimension
        return torch.cat(conv_results, dim=1)
    
class UNet_DynamicPyramid(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_splits=4, init_features=32):
        super(UNet_DynamicPyramid, self).__init__()
        self.num_splits = num_splits
        features = init_features
        self.encoder1 = nn.Sequential(nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,),
                            nn.BatchNorm2d(num_features=features), 
                             nn.ReLU(inplace=True),
                             DynamicSplConv2d(
                            in_channels=features,
                            out_channels=features, num_splits=self.num_splits), nn.BatchNorm2d(num_features=features),nn.ReLU(inplace=True),)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.block(features, features*2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.block(features*2, features*4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.block(features*4, features*8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = self.block(features*8, features*16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = self.block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = self.block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = self.block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = self.block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        dec1 = self.conv(dec1)       
        return dec1

    def block(self, in_channels, features):
        return nn.Sequential(DynamicSplConv2d(
                            in_channels=in_channels,
                            out_channels=features, num_splits=self.num_splits),
                            nn.BatchNorm2d(num_features=features), 
                             nn.ReLU(inplace=True),
                             DynamicSplConv2d(
                            in_channels=features,
                            out_channels=features, num_splits=self.num_splits), nn.BatchNorm2d(num_features=features),nn.ReLU(inplace=True),)
# import pytorch_model_summary
# print(pytorch_model_summary.summary(UNet_DynamicPyramid(1),torch.rand((1, 1, 512, 512))))
