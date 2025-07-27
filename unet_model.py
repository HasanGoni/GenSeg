"""
Robust UNet Implementation for Image Segmentation
Author: Expert Python Programmer
Purpose: Production-ready UNet for custom datasets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DoubleConv(nn.Module):
    """
    Double Convolution block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    This is the basic building block of UNet
    """
    
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        
        # Use bilinear interpolation or transposed convolutions for upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle input size mismatch (padding if necessary)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet Architecture for Image Segmentation
    
    Args:
        n_channels: Number of input channels (e.g., 3 for RGB, 1 for grayscale)
        n_classes: Number of output classes for segmentation
        bilinear: Use bilinear interpolation for upsampling (vs transposed conv)
    """
    
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        # Validate inputs
        if n_channels <= 0:
            raise ValueError(f"n_channels must be positive, got {n_channels}")
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")
        
        # Encoder (downsampling path)
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        # Decoder (upsampling path)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"UNet initialized with {n_channels} input channels and {n_classes} output classes")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the UNet"""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input tensor (B, C, H, W), got {x.dim()}D")
        if x.size(1) != self.n_channels:
            raise ValueError(f"Expected {self.n_channels} input channels, got {x.size(1)}")
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
    
    def _initialize_weights(self):
        """Initialize model weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def get_model_size(self) -> int:
        """Get total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save_checkpoint(self, filepath: str, epoch: int, optimizer_state: dict = None, 
                       loss: float = None, metrics: dict = None):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'model_config': {
                'n_channels': self.n_channels,
                'n_classes': self.n_classes,
                'bilinear': self.bilinear
            },
            'loss': loss,
            'metrics': metrics
        }
        
        if optimizer_state:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    @classmethod
    def load_checkpoint(cls, filepath: str, device: str = 'cpu'):
        """Load model from checkpoint"""
        checkpoint = torch.load(filepath, map_location=device)
        config = checkpoint['model_config']
        
        model = cls(
            n_channels=config['n_channels'],
            n_classes=config['n_classes'],
            bilinear=config['bilinear']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {filepath}")
        
        return model, checkpoint


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Example: 3-channel input (RGB), 2-class output (binary segmentation)
    model = UNet(n_channels=3, n_classes=2, bilinear=True).to(device)
    
    # Test with dummy data
    x = torch.randn(1, 3, 256, 256).to(device)
    output = model(x)
    
    print(f"Model parameters: {model.get_model_size():,}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("Model test passed!")