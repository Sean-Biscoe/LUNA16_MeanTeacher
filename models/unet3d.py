import torch
import torch.nn as nn

class ResidualBlock3D(nn.Module):
    # A simple 3D residual block with two convolutional layers and a skip connection
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch)
        )
        # If input and output channels differ, use a 1x1 convolution to match dimensions for the skip connection
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=1),
                nn.BatchNorm3d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv(x)
        out += residual
        return self.relu(out)

class LUNA16_UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_ch=32): # Start with 32 for better feature extraction
        super().__init__()
        
        # Encoder
        self.enc1 = ResidualBlock3D(in_channels, base_ch)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = ResidualBlock3D(base_ch, base_ch * 2)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = ResidualBlock3D(base_ch * 2, base_ch * 4)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock3D(base_ch * 4, base_ch * 8)

        # Decoder
        self.up3 = nn.ConvTranspose3d(base_ch * 8, base_ch * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock3D(base_ch * 8, base_ch * 4) # 4 (up) + 4 (skip)

        self.up2 = nn.ConvTranspose3d(base_ch * 4, base_ch * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock3D(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock3D(base_ch * 2, base_ch)

        self.final = nn.Conv3d(base_ch, out_channels, kernel_size=1)

    # The forward pass through the UNet architecture, with skip connections between encoder and decoder
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        
        b = self.bottleneck(self.pool3(e3))
        
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([e3, d3], dim=1)) # Concatenate along channel dim
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([e2, d2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([e1, d1], dim=1))
        
        return self.final(d1)
