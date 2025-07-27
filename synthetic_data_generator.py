"""
Synthetic Data Generation Pipeline for Image Segmentation
Implements multiple generative models to augment small datasets

Author: Expert Python Programmer
Purpose: Generate synthetic image-mask pairs to improve segmentation performance
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Generator(nn.Module):
    """
    Pix2Pix Generator for mask-to-image translation
    U-Net architecture with skip connections
    """
    
    def __init__(self, input_nc: int = 1, output_nc: int = 3, ngf: int = 64):
        super(Generator, self).__init__()
        
        # Encoder (downsampling)
        self.down1 = self._make_layer(input_nc, ngf, normalize=False)
        self.down2 = self._make_layer(ngf, ngf * 2)
        self.down3 = self._make_layer(ngf * 2, ngf * 4)
        self.down4 = self._make_layer(ngf * 4, ngf * 8)
        self.down5 = self._make_layer(ngf * 8, ngf * 8)
        self.down6 = self._make_layer(ngf * 8, ngf * 8)
        self.down7 = self._make_layer(ngf * 8, ngf * 8)
        self.down8 = self._make_layer(ngf * 8, ngf * 8, normalize=False)
        
        # Decoder (upsampling)
        self.up1 = self._make_uplayer(ngf * 8, ngf * 8, dropout=True)
        self.up2 = self._make_uplayer(ngf * 16, ngf * 8, dropout=True)
        self.up3 = self._make_uplayer(ngf * 16, ngf * 8, dropout=True)
        self.up4 = self._make_uplayer(ngf * 16, ngf * 8)
        self.up5 = self._make_uplayer(ngf * 16, ngf * 4)
        self.up6 = self._make_uplayer(ngf * 8, ngf * 2)
        self.up7 = self._make_uplayer(ngf * 4, ngf)
        self.up8 = nn.ConvTranspose2d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1)
    
    def _make_layer(self, in_channels: int, out_channels: int, normalize: bool = True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
    
    def _make_uplayer(self, in_channels: int, out_channels: int, dropout: bool = False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        
        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], 1))
        u3 = self.up3(torch.cat([u2, d6], 1))
        u4 = self.up4(torch.cat([u3, d5], 1))
        u5 = self.up5(torch.cat([u4, d4], 1))
        u6 = self.up6(torch.cat([u5, d3], 1))
        u7 = self.up7(torch.cat([u6, d2], 1))
        u8 = self.up8(torch.cat([u7, d1], 1))
        
        return torch.tanh(u8)


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for realistic image generation
    """
    
    def __init__(self, input_nc: int = 4, ndf: int = 64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class VAEGenerator(nn.Module):
    """
    Variational Autoencoder for generating diverse synthetic samples
    """
    
    def __init__(self, input_nc: int = 1, output_nc: int = 3, latent_dim: int = 512):
        super(VAEGenerator, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        
        # Latent space
        self.fc_mu = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(256 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 16 * 16)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_nc, 4, 2, 1),
            nn.Tanh()
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(h.size(0), 256, 16, 16)
        return self.decoder(h)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class SyntheticDataset(Dataset):
    """Dataset class for loading real image-mask pairs"""
    
    def __init__(self, images_dir: str, masks_dir: str, image_size: Tuple[int, int] = (256, 256)):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = image_size
        
        # Find all image files
        self.image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tiff']:
            self.image_files.extend(list(self.images_dir.glob(f'*{ext}')))
        
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])
        
        self.normalize_image = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        self.normalize_mask = transforms.Normalize([0.5], [0.5])
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.masks_dir / f"{img_path.stem}_mask{img_path.suffix}"
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Transform
        image = self.transform(image)
        mask = self.transform(mask)
        
        # Normalize
        image = self.normalize_image(image)
        mask = self.normalize_mask(mask)
        
        return {
            'image': image,
            'mask': mask,
            'path': str(img_path)
        }


class SyntheticDataGenerator:
    """
    Main class for generating synthetic image-mask pairs
    """
    
    def __init__(
        self,
        model_type: str = 'pix2pix',
        device: str = 'cuda',
        image_size: Tuple[int, int] = (256, 256),
        latent_dim: int = 512
    ):
        self.model_type = model_type
        self.device = device
        self.image_size = image_size
        self.latent_dim = latent_dim
        
        # Initialize models
        if model_type == 'pix2pix':
            self.generator = Generator(input_nc=1, output_nc=3).to(device)
            self.discriminator = Discriminator(input_nc=4).to(device)
        elif model_type == 'vae':
            self.generator = VAEGenerator(input_nc=1, output_nc=3, latent_dim=latent_dim).to(device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Initialized {model_type} generator on {device}")
    
    def train_pix2pix(
        self,
        dataloader: DataLoader,
        num_epochs: int = 100,
        lr: float = 0.0002,
        lambda_l1: float = 100.0,
        save_dir: str = 'synthetic_models'
    ):
        """Train Pix2Pix model for mask-to-image translation"""
        
        # Optimizers
        optimizer_G = optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizer_D = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
        
        # Loss functions
        criterion_GAN = nn.BCELoss()
        criterion_L1 = nn.L1Loss()
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting Pix2Pix training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in progress_bar:
                real_images = batch['image'].to(self.device)
                real_masks = batch['mask'].to(self.device)
                batch_size = real_images.size(0)
                
                # Real and fake labels
                real_labels = torch.ones(batch_size, 1, 30, 30).to(self.device)
                fake_labels = torch.zeros(batch_size, 1, 30, 30).to(self.device)
                
                # Train Generator
                optimizer_G.zero_grad()
                
                fake_images = self.generator(real_masks)
                fake_pairs = torch.cat([real_masks, fake_images], 1)
                pred_fake = self.discriminator(fake_pairs)
                
                loss_G_GAN = criterion_GAN(pred_fake, real_labels)
                loss_G_L1 = criterion_L1(fake_images, real_images) * lambda_l1
                loss_G = loss_G_GAN + loss_G_L1
                
                loss_G.backward()
                optimizer_G.step()
                
                # Train Discriminator
                optimizer_D.zero_grad()
                
                # Real pairs
                real_pairs = torch.cat([real_masks, real_images], 1)
                pred_real = self.discriminator(real_pairs)
                loss_D_real = criterion_GAN(pred_real, real_labels)
                
                # Fake pairs
                fake_pairs = torch.cat([real_masks, fake_images.detach()], 1)
                pred_fake = self.discriminator(fake_pairs)
                loss_D_fake = criterion_GAN(pred_fake, fake_labels)
                
                loss_D = (loss_D_real + loss_D_fake) * 0.5
                loss_D.backward()
                optimizer_D.step()
                
                epoch_g_loss += loss_G.item()
                epoch_d_loss += loss_D.item()
                
                progress_bar.set_postfix({
                    'G_Loss': f'{loss_G.item():.4f}',
                    'D_Loss': f'{loss_D.item():.4f}'
                })
            
            # Save checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save(self.generator.state_dict(), 
                          save_path / f'generator_epoch_{epoch+1}.pth')
                torch.save(self.discriminator.state_dict(), 
                          save_path / f'discriminator_epoch_{epoch+1}.pth')
            
            logger.info(f"Epoch {epoch+1}: G_Loss={epoch_g_loss/len(dataloader):.4f}, "
                       f"D_Loss={epoch_d_loss/len(dataloader):.4f}")
        
        # Save final models
        torch.save(self.generator.state_dict(), save_path / 'generator_final.pth')
        torch.save(self.discriminator.state_dict(), save_path / 'discriminator_final.pth')
        
        logger.info("Pix2Pix training completed")
    
    def train_vae(
        self,
        dataloader: DataLoader,
        num_epochs: int = 100,
        lr: float = 0.001,
        beta: float = 1.0,
        save_dir: str = 'synthetic_models'
    ):
        """Train VAE model for diverse sample generation"""
        
        optimizer = optim.Adam(self.generator.parameters(), lr=lr)
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting VAE training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch in progress_bar:
                real_masks = batch['mask'].to(self.device)
                
                optimizer.zero_grad()
                
                recon, mu, logvar = self.generator(real_masks)
                
                # VAE loss
                recon_loss = F.mse_loss(recon, real_masks, reduction='sum')
                kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + beta * kld_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Save checkpoints
            if (epoch + 1) % 10 == 0:
                torch.save(self.generator.state_dict(), 
                          save_path / f'vae_epoch_{epoch+1}.pth')
            
            logger.info(f"Epoch {epoch+1}: Loss={epoch_loss/len(dataloader):.4f}")
        
        # Save final model
        torch.save(self.generator.state_dict(), save_path / 'vae_final.pth')
        
        logger.info("VAE training completed")
    
    def generate_synthetic_data(
        self,
        input_masks: List[torch.Tensor],
        num_variations: int = 5,
        output_dir: str = 'synthetic_data',
        model_path: Optional[str] = None
    ) -> List[Dict[str, np.ndarray]]:
        """
        Generate synthetic image-mask pairs
        
        Args:
            input_masks: List of mask tensors to generate images from
            num_variations: Number of variations per mask
            output_dir: Directory to save generated data
            model_path: Path to trained generator model
            
        Returns:
            List of generated image-mask pairs
        """
        if model_path:
            self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
        
        self.generator.eval()
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        generated_data = []
        
        logger.info(f"Generating synthetic data with {num_variations} variations per mask")
        
        with torch.no_grad():
            for mask_idx, mask in enumerate(tqdm(input_masks, desc="Generating synthetic data")):
                mask = mask.unsqueeze(0).to(self.device)
                
                for var_idx in range(num_variations):
                    if self.model_type == 'pix2pix':
                        # Add noise for variation
                        noise = torch.randn_like(mask) * 0.1
                        noisy_mask = torch.clamp(mask + noise, -1, 1)
                        generated_image = self.generator(noisy_mask)
                    
                    elif self.model_type == 'vae':
                        # Sample from latent space for variation
                        z = torch.randn(1, self.latent_dim).to(self.device)
                        generated_image = self.generator.decode(z)
                    
                    # Convert to numpy
                    gen_img = generated_image.squeeze().cpu().numpy()
                    gen_mask = mask.squeeze().cpu().numpy()
                    
                    # Denormalize
                    gen_img = (gen_img + 1) / 2  # [-1, 1] -> [0, 1]
                    gen_mask = (gen_mask + 1) / 2
                    
                    # Convert to uint8
                    gen_img = (gen_img * 255).astype(np.uint8)
                    gen_mask = (gen_mask * 255).astype(np.uint8)
                    
                    # Save files
                    img_name = f"synthetic_img_{mask_idx:04d}_{var_idx:02d}.png"
                    mask_name = f"synthetic_img_{mask_idx:04d}_{var_idx:02d}_mask.png"
                    
                    if len(gen_img.shape) == 3:
                        gen_img = np.transpose(gen_img, (1, 2, 0))
                    
                    cv2.imwrite(str(output_path / img_name), 
                               cv2.cvtColor(gen_img, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(output_path / mask_name), gen_mask)
                    
                    generated_data.append({
                        'image': gen_img,
                        'mask': gen_mask,
                        'image_path': str(output_path / img_name),
                        'mask_path': str(output_path / mask_name)
                    })
        
        logger.info(f"Generated {len(generated_data)} synthetic samples")
        return generated_data
    
    def load_pretrained_model(self, model_path: str):
        """Load a pretrained generator model"""
        self.generator.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"Loaded pretrained model from {model_path}")
    
    def visualize_generation(
        self,
        input_mask: torch.Tensor,
        num_samples: int = 4,
        save_path: Optional[str] = None
    ):
        """Visualize generated samples for a given mask"""
        self.generator.eval()
        
        with torch.no_grad():
            mask = input_mask.unsqueeze(0).to(self.device)
            
            fig, axes = plt.subplots(2, num_samples + 1, figsize=(15, 6))
            
            # Show original mask
            mask_display = (mask.squeeze().cpu().numpy() + 1) / 2
            axes[0, 0].imshow(mask_display, cmap='gray')
            axes[0, 0].set_title('Input Mask')
            axes[0, 0].axis('off')
            axes[1, 0].axis('off')
            
            # Generate variations
            for i in range(num_samples):
                if self.model_type == 'pix2pix':
                    noise = torch.randn_like(mask) * 0.1
                    noisy_mask = torch.clamp(mask + noise, -1, 1)
                    generated = self.generator(noisy_mask)
                elif self.model_type == 'vae':
                    z = torch.randn(1, self.latent_dim).to(self.device)
                    generated = self.generator.decode(z)
                
                gen_img = generated.squeeze().cpu().numpy()
                gen_img = (gen_img + 1) / 2
                
                if len(gen_img.shape) == 3:
                    gen_img = np.transpose(gen_img, (1, 2, 0))
                
                axes[0, i + 1].imshow(gen_img)
                axes[0, i + 1].set_title(f'Generated {i+1}')
                axes[0, i + 1].axis('off')
                
                # Show difference from original
                if i == 0:
                    axes[1, i + 1].imshow(gen_img)
                    axes[1, i + 1].set_title('Reference')
                else:
                    diff = np.abs(gen_img - axes[0, 1].images[0].get_array())
                    axes[1, i + 1].imshow(diff, cmap='hot')
                    axes[1, i + 1].set_title(f'Difference {i+1}')
                axes[1, i + 1].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to {save_path}")
            
            plt.show()


def create_synthetic_dataloader(
    images_dir: str,
    masks_dir: str,
    batch_size: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4
) -> DataLoader:
    """Create dataloader for synthetic data generation training"""
    
    dataset = SyntheticDataset(images_dir, masks_dir, image_size)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    logger.info(f"Created synthetic dataloader with {len(dataset)} samples")
    return dataloader


if __name__ == "__main__":
    # Example usage
    print("Synthetic Data Generator loaded successfully!")
    print("Example usage:")
    print("""
    from synthetic_data_generator import SyntheticDataGenerator, create_synthetic_dataloader
    
    # Create dataloader from small dataset
    dataloader = create_synthetic_dataloader(
        images_dir='data/train/images',
        masks_dir='data/train/masks',
        batch_size=4
    )
    
    # Initialize generator
    generator = SyntheticDataGenerator(model_type='pix2pix', device='cuda')
    
    # Train the generator
    generator.train_pix2pix(dataloader, num_epochs=100)
    
    # Generate synthetic data
    masks = [torch.randn(1, 256, 256) for _ in range(10)]  # Example masks
    synthetic_data = generator.generate_synthetic_data(
        input_masks=masks,
        num_variations=5,
        output_dir='synthetic_data'
    )
    
    print(f"Generated {len(synthetic_data)} synthetic samples")
    """)