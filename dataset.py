"""
Custom Dataset Implementation for UNet Training
Handles image and mask loading with robust error handling and data augmentation
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Optional, List, Callable, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """
    Custom Dataset for Image Segmentation
    
    Supports:
    - Multiple image formats (jpg, png, tiff)
    - Data augmentation with albumentations
    - Proper error handling and validation
    - Memory-efficient loading
    """
    
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        transform: Optional[Callable] = None,
        image_size: Tuple[int, int] = (256, 256),
        mask_suffix: str = "_mask",
        image_extensions: List[str] = None,
        validate_data: bool = True
    ):
        """
        Args:
            images_dir: Directory containing input images
            masks_dir: Directory containing corresponding masks
            transform: Data augmentation transforms
            image_size: Target size for resizing (H, W)
            mask_suffix: Suffix to identify mask files
            image_extensions: Supported image file extensions
            validate_data: Whether to validate data integrity on initialization
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.image_size = image_size
        self.mask_suffix = mask_suffix
        
        if image_extensions is None:
            self.image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        else:
            self.image_extensions = image_extensions
        
        # Validate directories
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {masks_dir}")
        
        # Find all image files
        self.image_files = self._find_image_files()
        
        if len(self.image_files) == 0:
            raise ValueError(f"No valid image files found in {images_dir}")
        
        # Validate data integrity if requested
        if validate_data:
            self._validate_dataset()
        
        logger.info(f"Dataset initialized with {len(self.image_files)} samples")
    
    def _find_image_files(self) -> List[Path]:
        """Find all valid image files in the images directory"""
        image_files = []
        
        for ext in self.image_extensions:
            pattern = f"*{ext}"
            files = list(self.images_dir.glob(pattern))
            image_files.extend(files)
        
        # Sort for consistent ordering
        return sorted(image_files)
    
    def _get_mask_path(self, image_path: Path) -> Path:
        """Get corresponding mask path for an image"""
        # Remove extension and add mask suffix
        base_name = image_path.stem
        if not base_name.endswith(self.mask_suffix):
            mask_name = f"{base_name}{self.mask_suffix}{image_path.suffix}"
        else:
            mask_name = f"{base_name}{image_path.suffix}"
        
        return self.masks_dir / mask_name
    
    def _validate_dataset(self):
        """Validate that all images have corresponding masks"""
        missing_masks = []
        
        for img_path in self.image_files:
            mask_path = self._get_mask_path(img_path)
            if not mask_path.exists():
                missing_masks.append(str(mask_path))
        
        if missing_masks:
            logger.warning(f"Found {len(missing_masks)} images without corresponding masks")
            # Remove images without masks
            self.image_files = [
                img for img in self.image_files 
                if self._get_mask_path(img).exists()
            ]
            
        logger.info(f"Validation complete. {len(self.image_files)} valid samples found")
    
    def _load_image(self, path: Path) -> np.ndarray:
        """Load and preprocess image"""
        try:
            # Use OpenCV for better performance with large images
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError(f"Failed to load image: {path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if needed
            if image.shape[:2] != self.image_size:
                image = cv2.resize(image, self.image_size[::-1], interpolation=cv2.INTER_LINEAR)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {path}: {e}")
            raise
    
    def _load_mask(self, path: Path) -> np.ndarray:
        """Load and preprocess mask"""
        try:
            # Load mask as grayscale
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Failed to load mask: {path}")
            
            # Resize if needed
            if mask.shape != self.image_size:
                mask = cv2.resize(mask, self.image_size[::-1], interpolation=cv2.INTER_NEAREST)
            
            # Ensure binary mask (0 and 1)
            mask = (mask > 0).astype(np.uint8)
            
            return mask
            
        except Exception as e:
            logger.error(f"Error loading mask {path}: {e}")
            raise
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset"""
        if idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Load image and mask
        img_path = self.image_files[idx]
        mask_path = self._get_mask_path(img_path)
        
        image = self._load_image(img_path)
        mask = self._load_mask(mask_path)
        
        # Apply transforms if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Ensure tensors are in correct format
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        
        return {
            'image': image,
            'mask': mask,
            'image_path': str(img_path),
            'mask_path': str(mask_path)
        }
    
    def get_sample_weights(self) -> torch.Tensor:
        """Calculate sample weights for balanced training"""
        # This is a placeholder - implement based on your specific needs
        # For example, calculate weights based on class distribution
        return torch.ones(len(self))
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution across the dataset"""
        class_counts = {}
        
        for idx in range(len(self)):
            sample = self.__getitem__(idx)
            mask = sample['mask']
            unique_classes = torch.unique(mask)
            
            for cls in unique_classes:
                cls_id = cls.item()
                if cls_id not in class_counts:
                    class_counts[cls_id] = 0
                class_counts[cls_id] += (mask == cls).sum().item()
        
        return class_counts


def get_augmentation_pipeline(image_size: Tuple[int, int], is_training: bool = True) -> A.Compose:
    """
    Get augmentation pipeline for training or validation
    
    Args:
        image_size: Target image size (H, W)
        is_training: Whether to apply training augmentations
    """
    if is_training:
        transforms = [
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=3),
                A.MotionBlur(blur_limit=3),
            ], p=0.3),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.ElasticTransform(p=0.3),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    
    return A.Compose(transforms)


def create_dataloaders(
    train_images_dir: str,
    train_masks_dir: str,
    val_images_dir: str,
    val_masks_dir: str,
    batch_size: int = 4,
    image_size: Tuple[int, int] = (256, 256),
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation dataloaders
    
    Args:
        train_images_dir: Training images directory
        train_masks_dir: Training masks directory
        val_images_dir: Validation images directory
        val_masks_dir: Validation masks directory
        batch_size: Batch size for training
        image_size: Target image size
        num_workers: Number of worker processes for data loading
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Create datasets
    train_dataset = SegmentationDataset(
        images_dir=train_images_dir,
        masks_dir=train_masks_dir,
        transform=get_augmentation_pipeline(image_size, is_training=True),
        image_size=image_size,
        **kwargs
    )
    
    val_dataset = SegmentationDataset(
        images_dir=val_images_dir,
        masks_dir=val_masks_dir,
        transform=get_augmentation_pipeline(image_size, is_training=False),
        image_size=image_size,
        **kwargs
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    logger.info(f"Created dataloaders - Train: {len(train_dataset)} samples, "
                f"Val: {len(val_dataset)} samples")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage and testing
    print("Testing dataset implementation...")
    
    # This would fail unless you have actual data directories
    # Uncomment and modify paths when you have real data
    """
    dataset = SegmentationDataset(
        images_dir="data/train/images",
        masks_dir="data/train/masks",
        transform=get_augmentation_pipeline((256, 256), is_training=True)
    )
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Mask shape: {sample['mask'].shape}")
    print(f"Unique mask values: {torch.unique(sample['mask'])}")
    """
    
    print("Dataset module loaded successfully!")