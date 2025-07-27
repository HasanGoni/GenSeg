"""
Robust Training Pipeline for UNet Model
Includes metrics calculation, early stopping, model checkpointing, and logging
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from sklearn.metrics import jaccard_score, f1_score
from tqdm import tqdm
import json

from unet_model import UNet

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Calculate various segmentation metrics"""
    
    @staticmethod
    def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Calculate Dice coefficient (F1 score for segmentation)"""
        pred = torch.sigmoid(pred) if pred.max() > 1 else pred
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
        return dice.item()
    
    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
        """Calculate Intersection over Union (IoU)"""
        pred = torch.sigmoid(pred) if pred.max() > 1 else pred
        pred = (pred > 0.5).float()
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + smooth) / (union + smooth)
        return iou.item()
    
    @staticmethod
    def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate pixel-wise accuracy"""
        pred = torch.sigmoid(pred) if pred.max() > 1 else pred
        pred = (pred > 0.5).float()
        
        correct = (pred == target).float().sum()
        total = target.numel()
        return (correct / total).item()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save the best model weights"""
        self.best_weights = model.state_dict().copy()


class UNetTrainer:
    """
    Comprehensive trainer for UNet model with all production features
    """
    
    def __init__(
        self,
        model: UNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        early_stopping_patience: int = 15,
        save_every_n_epochs: int = 5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        self.metrics = SegmentationMetrics()
        self.save_every_n_epochs = save_every_n_epochs
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_loss = float('inf')
        
        logger.info(f"Trainer initialized - Device: {device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1} [Train]')
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss
            if self.model.n_classes == 1:
                # Binary segmentation
                loss = self.criterion(outputs.squeeze(1), masks.float())
                pred_masks = outputs.squeeze(1)
            else:
                # Multi-class segmentation
                loss = self.criterion(outputs, masks)
                pred_masks = torch.softmax(outputs, dim=1)[:, 1]  # Assuming class 1 is foreground
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                dice = self.metrics.dice_coefficient(pred_masks, masks.float())
                iou = self.metrics.iou_score(pred_masks, masks.float())
                accuracy = self.metrics.pixel_accuracy(pred_masks, masks.float())
            
            # Update running averages
            total_loss += loss.item()
            total_dice += dice
            total_iou += iou
            total_accuracy += accuracy
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{dice:.4f}',
                'IoU': f'{iou:.4f}'
            })
        
        # Calculate average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
            'iou': total_iou / num_batches,
            'accuracy': total_accuracy / num_batches
        }
        
        return avg_metrics
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_dice = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        num_batches = len(self.val_loader)
        
        progress_bar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch + 1} [Val]')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                if self.model.n_classes == 1:
                    loss = self.criterion(outputs.squeeze(1), masks.float())
                    pred_masks = outputs.squeeze(1)
                else:
                    loss = self.criterion(outputs, masks)
                    pred_masks = torch.softmax(outputs, dim=1)[:, 1]
                
                # Calculate metrics
                dice = self.metrics.dice_coefficient(pred_masks, masks.float())
                iou = self.metrics.iou_score(pred_masks, masks.float())
                accuracy = self.metrics.pixel_accuracy(pred_masks, masks.float())
                
                # Update running averages
                total_loss += loss.item()
                total_dice += dice
                total_iou += iou
                total_accuracy += accuracy
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Dice': f'{dice:.4f}',
                    'IoU': f'{iou:.4f}'
                })
        
        # Calculate average metrics
        avg_metrics = {
            'loss': total_loss / num_batches,
            'dice': total_dice / num_batches,
            'iou': total_iou / num_batches,
            'accuracy': total_accuracy / num_batches
        }
        
        return avg_metrics
    
    def log_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to tensorboard and console"""
        # Log to tensorboard
        for metric_name, value in train_metrics.items():
            self.writer.add_scalar(f'Train/{metric_name}', value, self.current_epoch)
        
        for metric_name, value in val_metrics.items():
            self.writer.add_scalar(f'Validation/{metric_name}', value, self.current_epoch)
        
        # Log learning rate
        if self.scheduler:
            current_lr = self.scheduler.get_last_lr()[0]
            self.writer.add_scalar('Learning_Rate', current_lr, self.current_epoch)
        
        # Console logging
        logger.info(f"Epoch {self.current_epoch + 1}")
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                   f"Dice: {train_metrics['dice']:.4f}, "
                   f"IoU: {train_metrics['iou']:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                   f"Dice: {val_metrics['dice']:.4f}, "
                   f"IoU: {val_metrics['iou']:.4f}")
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_data = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss,
            'model_config': {
                'n_channels': self.model.n_channels,
                'n_classes': self.model.n_classes,
                'bilinear': self.model.bilinear
            }
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch + 1}.pth'
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint_data, best_path)
            logger.info(f"New best model saved with validation loss: {metrics['loss']:.4f}")
    
    def train(self, num_epochs: int) -> Dict[str, List[float]]:
        """Main training loop"""
        logger.info(f"Starting training for {num_epochs} epochs")
        start_time = time.time()
        
        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch
                
                # Train and validate
                train_metrics = self.train_epoch()
                val_metrics = self.validate_epoch()
                
                # Store metrics
                self.train_losses.append(train_metrics['loss'])
                self.val_losses.append(val_metrics['loss'])
                self.train_metrics.append(train_metrics)
                self.val_metrics.append(val_metrics)
                
                # Log metrics
                self.log_metrics(train_metrics, val_metrics)
                
                # Update learning rate
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                
                # Check for best model
                is_best = val_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_metrics['loss']
                
                # Save checkpoint
                if (epoch + 1) % self.save_every_n_epochs == 0 or is_best:
                    self.save_checkpoint(val_metrics, is_best)
                
                # Early stopping check
                if self.early_stopping(val_metrics['loss'], self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise
        
        finally:
            # Save final checkpoint
            self.save_checkpoint(val_metrics if 'val_metrics' in locals() else {})
            
            # Save training history
            history = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics
            }
            
            history_path = self.checkpoint_dir / 'training_history.json'
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
            
            self.writer.close()
            
            total_time = time.time() - start_time
            logger.info(f"Training completed in {total_time:.2f} seconds")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }


def create_loss_function(n_classes: int, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
    """Create appropriate loss function for segmentation task"""
    if n_classes == 1:
        # Binary segmentation - use BCEWithLogitsLoss
        return nn.BCEWithLogitsLoss(pos_weight=class_weights)
    else:
        # Multi-class segmentation - use CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=class_weights)


def create_optimizer(
    model: nn.Module,
    optimizer_name: str = 'adam',
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    **kwargs
) -> optim.Optimizer:
    """Create optimizer for training"""
    optimizers = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop
    }
    
    if optimizer_name.lower() not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    optimizer_class = optimizers[optimizer_name.lower()]
    
    if optimizer_name.lower() == 'sgd':
        return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay, 
                             momentum=kwargs.get('momentum', 0.9))
    else:
        return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_name: str = 'plateau',
    **kwargs
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler"""
    if scheduler_name is None:
        return None
    
    schedulers = {
        'plateau': lambda: optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        ),
        'cosine': lambda: optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get('T_max', 50)
        ),
        'step': lambda: optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get('step_size', 10), gamma=0.1
        ),
        'exponential': lambda: optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=kwargs.get('gamma', 0.95)
        )
    }
    
    if scheduler_name.lower() not in schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    
    return schedulers[scheduler_name.lower()]()


if __name__ == "__main__":
    # Example training setup
    print("Trainer module loaded successfully!")
    print("Example usage:")
    print("""
    from unet_model import UNet
    from dataset import create_dataloaders
    from trainer import UNetTrainer, create_loss_function, create_optimizer, create_scheduler
    
    # Create model
    model = UNet(n_channels=3, n_classes=1, bilinear=True)
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_images_dir='data/train/images',
        train_masks_dir='data/train/masks',
        val_images_dir='data/val/images',
        val_masks_dir='data/val/masks',
        batch_size=4
    )
    
    # Create training components
    criterion = create_loss_function(n_classes=1)
    optimizer = create_optimizer(model, 'adam', lr=1e-4)
    scheduler = create_scheduler(optimizer, 'plateau')
    
    # Create trainer
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler
    )
    
    # Train model
    history = trainer.train(num_epochs=100)
    """)