"""
Complete UNet Pipeline for Image Segmentation
Main script demonstrating the entire workflow from data loading to inference

Author: Expert Python Programmer
Purpose: Production-ready UNet implementation for custom datasets
"""

import os
import argparse
import logging
import torch
from pathlib import Path
import json
from typing import Dict, Any

# Import our custom modules
from unet_model import UNet
from dataset import create_dataloaders, SegmentationDataset
from trainer import UNetTrainer, create_loss_function, create_optimizer, create_scheduler
from inference import load_model_for_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unet_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories(base_dir: str = "unet_project"):
    """Create necessary directories for the project"""
    dirs = [
        f"{base_dir}/data/train/images",
        f"{base_dir}/data/train/masks", 
        f"{base_dir}/data/val/images",
        f"{base_dir}/data/val/masks",
        f"{base_dir}/data/test/images",
        f"{base_dir}/checkpoints",
        f"{base_dir}/logs",
        f"{base_dir}/predictions",
        f"{base_dir}/configs"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {dir_path}")
    
    return base_dir


def create_sample_config(config_path: str):
    """Create a sample configuration file"""
    config = {
        "model": {
            "n_channels": 3,
            "n_classes": 1,
            "bilinear": True
        },
        "training": {
            "batch_size": 4,
            "num_epochs": 100,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "optimizer": "adam",
            "scheduler": "plateau",
            "early_stopping_patience": 15
        },
        "data": {
            "image_size": [256, 256],
            "mask_suffix": "_mask",
            "num_workers": 4,
            "train_images_dir": "data/train/images",
            "train_masks_dir": "data/train/masks",
            "val_images_dir": "data/val/images",
            "val_masks_dir": "data/val/masks"
        },
        "inference": {
            "threshold": 0.5,
            "save_probability_maps": True,
            "save_overlays": True
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Sample configuration saved to: {config_path}")
    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file: {e}")
        raise


def train_model(config: Dict[str, Any], device: str = 'auto') -> str:
    """
    Train UNet model with the given configuration
    
    Args:
        config: Configuration dictionary
        device: Device to use for training
        
    Returns:
        Path to the best model checkpoint
    """
    logger.info("Starting model training...")
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create model
    model_config = config['model']
    model = UNet(
        n_channels=model_config['n_channels'],
        n_classes=model_config['n_classes'],
        bilinear=model_config['bilinear']
    )
    logger.info(f"Model created with {model.get_model_size():,} parameters")
    
    # Create data loaders
    data_config = config['data']
    train_config = config['training']
    
    try:
        train_loader, val_loader = create_dataloaders(
            train_images_dir=data_config['train_images_dir'],
            train_masks_dir=data_config['train_masks_dir'],
            val_images_dir=data_config['val_images_dir'],
            val_masks_dir=data_config['val_masks_dir'],
            batch_size=train_config['batch_size'],
            image_size=tuple(data_config['image_size']),
            num_workers=data_config['num_workers'],
            mask_suffix=data_config['mask_suffix']
        )
    except FileNotFoundError as e:
        logger.error(f"Data directories not found: {e}")
        logger.info("Please ensure your data is organized as follows:")
        logger.info("data/train/images/ - training images")
        logger.info("data/train/masks/ - training masks") 
        logger.info("data/val/images/ - validation images")
        logger.info("data/val/masks/ - validation masks")
        raise
    
    # Create training components
    criterion = create_loss_function(n_classes=model_config['n_classes'])
    optimizer = create_optimizer(
        model=model,
        optimizer_name=train_config['optimizer'],
        lr=train_config['learning_rate'],
        weight_decay=train_config['weight_decay']
    )
    scheduler = create_scheduler(
        optimizer=optimizer,
        scheduler_name=train_config['scheduler']
    )
    
    # Create trainer
    trainer = UNetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        early_stopping_patience=train_config['early_stopping_patience']
    )
    
    # Train model
    try:
        history = trainer.train(num_epochs=train_config['num_epochs'])
        best_model_path = "checkpoints/best_model.pth"
        logger.info(f"Training completed. Best model saved at: {best_model_path}")
        return best_model_path
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def run_inference(
    model_path: str, 
    input_path: str, 
    output_dir: str, 
    config: Dict[str, Any],
    device: str = 'auto'
):
    """
    Run inference on images using trained model
    
    Args:
        model_path: Path to trained model checkpoint
        input_path: Path to input image or directory
        output_dir: Directory to save predictions
        config: Configuration dictionary
        device: Device to use for inference
    """
    logger.info("Starting inference...")
    
    # Load model for inference
    inference_config = config['inference']
    data_config = config['data']
    
    predictor = load_model_for_inference(
        model_path=model_path,
        device=device,
        image_size=tuple(data_config['image_size']),
        threshold=inference_config['threshold']
    )
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check if input is file or directory
    input_path = Path(input_path)
    
    if input_path.is_file():
        # Single image inference
        logger.info(f"Processing single image: {input_path}")
        
        # Predict
        mask, prob_map = predictor.predict_single(str(input_path), return_probability=True)
        
        # Save results
        base_name = input_path.stem
        
        # Save mask
        mask_path = Path(output_dir) / f"{base_name}_mask.png"
        import cv2
        cv2.imwrite(str(mask_path), mask * 255)
        
        # Save probability map if requested
        if inference_config['save_probability_maps']:
            prob_path = Path(output_dir) / f"{base_name}_prob.png"
            cv2.imwrite(str(prob_path), (prob_map * 255).astype('uint8'))
        
        # Save overlay if requested
        if inference_config['save_overlays']:
            overlay = predictor.create_overlay(str(input_path), mask)
            overlay_path = Path(output_dir) / f"{base_name}_overlay.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        
        # Create visualization
        viz_path = Path(output_dir) / f"{base_name}_visualization.png"
        predictor.visualize_prediction(
            str(input_path), 
            save_path=str(viz_path), 
            show_plot=False
        )
        
        # Get and log statistics
        stats = predictor.get_prediction_stats(mask)
        logger.info(f"Prediction statistics for {input_path.name}:")
        logger.info(f"  Coverage: {stats['coverage_percentage']:.2f}%")
        logger.info(f"  Positive pixels: {stats['positive_pixels']:,}")
        
    elif input_path.is_dir():
        # Directory inference
        logger.info(f"Processing directory: {input_path}")
        
        predictor.predict_directory(
            input_dir=str(input_path),
            output_dir=output_dir,
            save_probability_maps=inference_config['save_probability_maps'],
            save_overlays=inference_config['save_overlays']
        )
        
    else:
        logger.error(f"Input path does not exist: {input_path}")
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    logger.info(f"Inference completed. Results saved to: {output_dir}")


def main():
    """Main function to run the UNet pipeline"""
    parser = argparse.ArgumentParser(description="UNet Pipeline for Image Segmentation")
    parser.add_argument("--mode", choices=["setup", "train", "inference"], required=True,
                       help="Mode to run: setup, train, or inference")
    parser.add_argument("--config", type=str, default="configs/config.json",
                       help="Path to configuration file")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                       help="Path to model checkpoint (for inference)")
    parser.add_argument("--input", type=str, 
                       help="Input image or directory path (for inference)")
    parser.add_argument("--output", type=str, default="predictions",
                       help="Output directory for predictions")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use for training/inference")
    
    args = parser.parse_args()
    
    try:
        if args.mode == "setup":
            # Setup project structure
            logger.info("Setting up project directories...")
            project_dir = setup_directories()
            config_path = f"{project_dir}/configs/config.json"
            create_sample_config(config_path)
            
            logger.info("Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Add your training data to:")
            logger.info(f"   - {project_dir}/data/train/images/")
            logger.info(f"   - {project_dir}/data/train/masks/")
            logger.info(f"   - {project_dir}/data/val/images/")
            logger.info(f"   - {project_dir}/data/val/masks/")
            logger.info("2. Adjust configuration if needed:")
            logger.info(f"   - {config_path}")
            logger.info("3. Run training:")
            logger.info(f"   python main.py --mode train --config {config_path}")
            
        elif args.mode == "train":
            # Train model
            if not os.path.exists(args.config):
                logger.error(f"Configuration file not found: {args.config}")
                logger.info("Run 'python main.py --mode setup' first to create the project structure")
                return
            
            config = load_config(args.config)
            best_model_path = train_model(config, args.device)
            
            logger.info("\nTraining completed successfully!")
            logger.info(f"Best model saved at: {best_model_path}")
            logger.info("Run inference with:")
            logger.info(f"python main.py --mode inference --model {best_model_path} --input path/to/image.jpg")
            
        elif args.mode == "inference":
            # Run inference
            if not args.input:
                logger.error("Input path is required for inference mode")
                return
            
            if not os.path.exists(args.model):
                logger.error(f"Model checkpoint not found: {args.model}")
                logger.info("Train a model first using: python main.py --mode train")
                return
            
            if not os.path.exists(args.config):
                logger.warning(f"Configuration file not found: {args.config}")
                logger.info("Using default configuration...")
                config = create_sample_config("temp_config.json")
                os.remove("temp_config.json")
            else:
                config = load_config(args.config)
            
            run_inference(args.model, args.input, args.output, config, args.device)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    print("=" * 80)
    print("UNet Image Segmentation Pipeline")
    print("Production-ready implementation for custom datasets")
    print("=" * 80)
    print()
    
    # Show usage examples if no arguments provided
    import sys
    if len(sys.argv) == 1:
        print("Usage Examples:")
        print()
        print("1. Setup project structure:")
        print("   python main.py --mode setup")
        print()
        print("2. Train model:")
        print("   python main.py --mode train --config configs/config.json")
        print()
        print("3. Run inference on single image:")
        print("   python main.py --mode inference --model checkpoints/best_model.pth --input image.jpg")
        print()
        print("4. Run inference on directory:")
        print("   python main.py --mode inference --model checkpoints/best_model.pth --input images/ --output results/")
        print()
        print("For detailed help:")
        print("   python main.py --help")
        print()
    else:
        main()