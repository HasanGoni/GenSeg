"""
Complete Generative AI Pipeline for Image Segmentation
Integrates synthetic data generation, model training, and feedback optimization

Author: Expert Python Programmer
Purpose: End-to-end pipeline from small dataset to optimized segmentation model
"""

import os
import argparse
import logging
import torch
import numpy as np
import cv2
from pathlib import Path
import json
import time
from typing import Dict, List, Any, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import our custom modules
from unet_model import UNet
from dataset import SegmentationDataset, create_dataloaders
from trainer import UNetTrainer, create_loss_function, create_optimizer, create_scheduler
from inference import load_model_for_inference
from synthetic_data_generator import SyntheticDataGenerator, create_synthetic_dataloader
from feedback_optimizer import FeedbackOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generative_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GenerativePipelineManager:
    """
    Main manager for the complete generative AI segmentation pipeline
    """
    
    def __init__(
        self,
        base_dir: str = "generative_segmentation_project",
        device: str = 'auto',
        config_path: Optional[str] = None
    ):
        self.base_dir = Path(base_dir)
        self.device = self._get_device(device)
        
        # Load or create configuration
        if config_path and Path(config_path).exists():
            self.config = self._load_config(config_path)
        else:
            self.config = self._create_default_config()
        
        # Initialize pipeline state
        self.pipeline_state = {
            'stage': 'initialization',
            'synthetic_generator': None,
            'segmentation_model': None,
            'feedback_optimizer': None,
            'performance_history': [],
            'optimization_rounds': 0
        }
        
        logger.info(f"Generative Pipeline Manager initialized on {self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for the pipeline"""
        return {
            "pipeline": {
                "stages": ["synthetic_generation", "initial_training", "feedback_optimization"],
                "max_optimization_rounds": 20,
                "patience": 10,
                "min_improvement": 0.01
            },
            "synthetic_generation": {
                "model_type": "pix2pix",
                "training_epochs": 50,
                "learning_rate": 0.0002,
                "lambda_l1": 100.0,
                "samples_per_mask": 5,
                "quality_threshold": 0.7
            },
            "segmentation": {
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
                }
            },
            "data": {
                "image_size": [256, 256],
                "mask_suffix": "_mask",
                "num_workers": 4,
                "augmentation_strength": "medium"
            },
            "feedback_optimization": {
                "optimization_patience": 8,
                "quality_weight": 0.4,
                "performance_weight": 0.6,
                "adaptation_rate": 0.1
            }
        }
    
    def setup_project_structure(self):
        """Create the complete project directory structure"""
        dirs = [
            f"{self.base_dir}/data/original/images",
            f"{self.base_dir}/data/original/masks",
            f"{self.base_dir}/data/synthetic/generated",
            f"{self.base_dir}/data/synthetic/optimized",
            f"{self.base_dir}/data/combined/train/images",
            f"{self.base_dir}/data/combined/train/masks",
            f"{self.base_dir}/data/combined/val/images",
            f"{self.base_dir}/data/combined/val/masks",
            f"{self.base_dir}/models/synthetic_generators",
            f"{self.base_dir}/models/segmentation_checkpoints",
            f"{self.base_dir}/models/optimized_models",
            f"{self.base_dir}/results/visualizations",
            f"{self.base_dir}/results/metrics",
            f"{self.base_dir}/results/optimization_logs",
            f"{self.base_dir}/configs",
            f"{self.base_dir}/logs"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = self.base_dir / "configs" / "pipeline_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Project structure created at {self.base_dir}")
        return str(config_path)
    
    def load_original_data(self, images_dir: str, masks_dir: str) -> List[Dict[str, np.ndarray]]:
        """Load original small dataset"""
        images_path = Path(images_dir)
        masks_path = Path(masks_dir)
        
        if not images_path.exists() or not masks_path.exists():
            raise FileNotFoundError(f"Data directories not found: {images_dir}, {masks_dir}")
        
        # Find all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.tiff']:
            image_files.extend(list(images_path.glob(f'*{ext}')))
        
        data_samples = []
        mask_suffix = self.config['data']['mask_suffix']
        
        for img_path in image_files:
            mask_path = masks_path / f"{img_path.stem}{mask_suffix}{img_path.suffix}"
            
            if mask_path.exists():
                # Load image and mask
                image = cv2.imread(str(img_path))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                
                # Resize to target size
                target_size = tuple(self.config['data']['image_size'])
                image = cv2.resize(image, target_size[::-1])
                mask = cv2.resize(mask, target_size[::-1])
                
                data_samples.append({
                    'image': image,
                    'mask': mask,
                    'image_path': str(img_path),
                    'mask_path': str(mask_path)
                })
        
        logger.info(f"Loaded {len(data_samples)} original samples")
        return data_samples
    
    def train_synthetic_generator(self, original_data: List[Dict[str, np.ndarray]]) -> SyntheticDataGenerator:
        """Train the synthetic data generator"""
        logger.info("Training synthetic data generator...")
        
        # Save original data temporarily for dataloader
        temp_dir = self.base_dir / "temp_original"
        temp_img_dir = temp_dir / "images"
        temp_mask_dir = temp_dir / "masks"
        temp_img_dir.mkdir(parents=True, exist_ok=True)
        temp_mask_dir.mkdir(parents=True, exist_ok=True)
        
        # Save images for dataloader
        for i, sample in enumerate(original_data):
            img_name = f"original_{i:04d}.png"
            mask_name = f"original_{i:04d}_mask.png"
            
            cv2.imwrite(str(temp_img_dir / img_name), 
                       cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(temp_mask_dir / mask_name), sample['mask'])
        
        # Create dataloader
        syn_config = self.config['synthetic_generation']
        dataloader = create_synthetic_dataloader(
            images_dir=str(temp_img_dir),
            masks_dir=str(temp_mask_dir),
            batch_size=min(4, len(original_data)),
            image_size=tuple(self.config['data']['image_size'])
        )
        
        # Initialize generator
        generator = SyntheticDataGenerator(
            model_type=syn_config['model_type'],
            device=self.device,
            image_size=tuple(self.config['data']['image_size'])
        )
        
        # Train generator
        save_dir = self.base_dir / "models" / "synthetic_generators"
        
        if syn_config['model_type'] == 'pix2pix':
            generator.train_pix2pix(
                dataloader=dataloader,
                num_epochs=syn_config['training_epochs'],
                lr=syn_config['learning_rate'],
                lambda_l1=syn_config['lambda_l1'],
                save_dir=str(save_dir)
            )
        elif syn_config['model_type'] == 'vae':
            generator.train_vae(
                dataloader=dataloader,
                num_epochs=syn_config['training_epochs'],
                lr=syn_config['learning_rate'],
                save_dir=str(save_dir)
            )
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(temp_dir)
        
        self.pipeline_state['synthetic_generator'] = generator
        self.pipeline_state['stage'] = 'synthetic_generation_complete'
        
        logger.info("Synthetic generator training completed")
        return generator
    
    def generate_synthetic_data(
        self,
        generator: SyntheticDataGenerator,
        original_data: List[Dict[str, np.ndarray]],
        output_dir: Optional[str] = None
    ) -> List[Dict[str, np.ndarray]]:
        """Generate synthetic data using trained generator"""
        
        if output_dir is None:
            output_dir = self.base_dir / "data" / "synthetic" / "generated"
        
        logger.info("Generating synthetic data...")
        
        # Convert masks to tensors
        masks = []
        for sample in original_data:
            mask_tensor = torch.from_numpy(sample['mask']).float() / 255.0
            mask_tensor = mask_tensor * 2.0 - 1.0  # Normalize to [-1, 1]
            masks.append(mask_tensor)
        
        # Generate synthetic data
        syn_config = self.config['synthetic_generation']
        synthetic_data = generator.generate_synthetic_data(
            input_masks=masks,
            num_variations=syn_config['samples_per_mask'],
            output_dir=str(output_dir)
        )
        
        logger.info(f"Generated {len(synthetic_data)} synthetic samples")
        return synthetic_data
    
    def combine_datasets(
        self,
        original_data: List[Dict[str, np.ndarray]],
        synthetic_data: List[Dict[str, np.ndarray]],
        train_ratio: float = 0.8
    ):
        """Combine original and synthetic data for training"""
        
        all_data = original_data + synthetic_data
        
        # Shuffle data
        import random
        random.shuffle(all_data)
        
        # Split into train/val
        split_idx = int(len(all_data) * train_ratio)
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        # Save combined dataset
        combined_dir = self.base_dir / "data" / "combined"
        
        # Save training data
        train_img_dir = combined_dir / "train" / "images"
        train_mask_dir = combined_dir / "train" / "masks"
        
        for i, sample in enumerate(train_data):
            img_name = f"combined_train_{i:06d}.png"
            mask_name = f"combined_train_{i:06d}_mask.png"
            
            cv2.imwrite(str(train_img_dir / img_name),
                       cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(train_mask_dir / mask_name), sample['mask'])
        
        # Save validation data
        val_img_dir = combined_dir / "val" / "images"
        val_mask_dir = combined_dir / "val" / "masks"
        
        for i, sample in enumerate(val_data):
            img_name = f"combined_val_{i:06d}.png"
            mask_name = f"combined_val_{i:06d}_mask.png"
            
            cv2.imwrite(str(val_img_dir / img_name),
                       cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(val_mask_dir / mask_name), sample['mask'])
        
        logger.info(f"Combined dataset created - Train: {len(train_data)}, Val: {len(val_data)}")
        
        return {
            'train_images_dir': str(train_img_dir),
            'train_masks_dir': str(train_mask_dir),
            'val_images_dir': str(val_img_dir),
            'val_masks_dir': str(val_mask_dir)
        }
    
    def train_segmentation_model(self, data_dirs: Dict[str, str]) -> UNet:
        """Train the segmentation model"""
        logger.info("Training segmentation model...")
        
        # Create model
        model_config = self.config['segmentation']['model']
        model = UNet(
            n_channels=model_config['n_channels'],
            n_classes=model_config['n_classes'],
            bilinear=model_config['bilinear']
        )
        
        # Create data loaders
        train_config = self.config['segmentation']['training']
        data_config = self.config['data']
        
        train_loader, val_loader = create_dataloaders(
            train_images_dir=data_dirs['train_images_dir'],
            train_masks_dir=data_dirs['train_masks_dir'],
            val_images_dir=data_dirs['val_images_dir'],
            val_masks_dir=data_dirs['val_masks_dir'],
            batch_size=train_config['batch_size'],
            image_size=tuple(data_config['image_size']),
            num_workers=data_config['num_workers'],
            mask_suffix=data_config['mask_suffix']
        )
        
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
        checkpoint_dir = self.base_dir / "models" / "segmentation_checkpoints"
        log_dir = self.base_dir / "logs" / "segmentation"
        
        trainer = UNetTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            early_stopping_patience=train_config['early_stopping_patience']
        )
        
        # Train model
        history = trainer.train(num_epochs=train_config['num_epochs'])
        
        self.pipeline_state['segmentation_model'] = model
        self.pipeline_state['stage'] = 'initial_training_complete'
        
        logger.info("Segmentation model training completed")
        return model
    
    def run_feedback_optimization(
        self,
        model: UNet,
        generator: SyntheticDataGenerator,
        original_data: List[Dict[str, np.ndarray]],
        test_data: List[Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Run feedback optimization loop"""
        logger.info("Starting feedback optimization...")
        
        # Initialize feedback optimizer
        opt_config = self.config['feedback_optimization']
        optimizer = FeedbackOptimizer(
            segmentation_model=model,
            synthetic_generator=generator,
            device=self.device,
            optimization_patience=opt_config['optimization_patience']
        )
        
        # Run optimization
        pipeline_config = self.config['pipeline']
        optimization_results = optimizer.optimize_synthetic_generation(
            real_data=original_data,
            test_data=test_data,
            max_optimization_rounds=pipeline_config['max_optimization_rounds'],
            synthetic_samples_per_round=len(original_data) * self.config['synthetic_generation']['samples_per_mask']
        )
        
        # Save optimization results
        results_dir = self.base_dir / "results" / "optimization_logs"
        results_path = results_dir / f"optimization_results_{int(time.time())}.json"
        optimizer.save_optimization_results(optimization_results, str(results_path))
        
        # Create visualization
        viz_path = self.base_dir / "results" / "visualizations" / f"optimization_progress_{int(time.time())}.png"
        optimizer.visualize_optimization_progress(optimization_results, str(viz_path))
        
        self.pipeline_state['feedback_optimizer'] = optimizer
        self.pipeline_state['stage'] = 'feedback_optimization_complete'
        self.pipeline_state['optimization_rounds'] = len(optimization_results['performance_history'])
        
        logger.info("Feedback optimization completed")
        return optimization_results
    
    def run_complete_pipeline(
        self,
        images_dir: str,
        masks_dir: str,
        test_images_dir: Optional[str] = None,
        test_masks_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the complete generative AI pipeline"""
        
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE GENERATIVE AI SEGMENTATION PIPELINE")
        logger.info("=" * 80)
        
        start_time = time.time()
        pipeline_results = {
            'stages_completed': [],
            'performance_metrics': {},
            'final_model_path': None,
            'synthetic_data_count': 0,
            'optimization_results': None
        }
        
        try:
            # Stage 1: Load original data
            logger.info("STAGE 1: Loading original data...")
            original_data = self.load_original_data(images_dir, masks_dir)
            
            if len(original_data) < 3:
                raise ValueError(f"Insufficient data: found {len(original_data)} samples, need at least 3")
            
            pipeline_results['stages_completed'].append('data_loading')
            pipeline_results['original_data_count'] = len(original_data)
            
            # Stage 2: Train synthetic generator
            logger.info("STAGE 2: Training synthetic data generator...")
            generator = self.train_synthetic_generator(original_data)
            pipeline_results['stages_completed'].append('synthetic_generator_training')
            
            # Stage 3: Generate synthetic data
            logger.info("STAGE 3: Generating synthetic data...")
            synthetic_data = self.generate_synthetic_data(generator, original_data)
            pipeline_results['stages_completed'].append('synthetic_data_generation')
            pipeline_results['synthetic_data_count'] = len(synthetic_data)
            
            # Stage 4: Combine datasets
            logger.info("STAGE 4: Combining datasets...")
            data_dirs = self.combine_datasets(original_data, synthetic_data)
            pipeline_results['stages_completed'].append('dataset_combination')
            
            # Stage 5: Train segmentation model
            logger.info("STAGE 5: Training segmentation model...")
            model = self.train_segmentation_model(data_dirs)
            pipeline_results['stages_completed'].append('segmentation_training')
            
            # Save initial model
            initial_model_path = self.base_dir / "models" / "optimized_models" / "initial_model.pth"
            model.save_checkpoint(
                str(initial_model_path),
                epoch=0,
                loss=0.0,
                metrics={'stage': 'initial_training'}
            )
            
            # Stage 6: Prepare test data
            if test_images_dir and test_masks_dir:
                test_data = self.load_original_data(test_images_dir, test_masks_dir)
            else:
                # Use a subset of original data as test
                test_size = max(1, len(original_data) // 4)
                test_data = original_data[-test_size:]
                logger.info(f"Using {len(test_data)} samples from original data as test set")
            
            # Stage 7: Feedback optimization
            logger.info("STAGE 6: Running feedback optimization...")
            optimization_results = self.run_feedback_optimization(
                model, generator, original_data, test_data
            )
            pipeline_results['stages_completed'].append('feedback_optimization')
            pipeline_results['optimization_results'] = optimization_results
            
            # Save final optimized model
            final_model_path = self.base_dir / "models" / "optimized_models" / "final_optimized_model.pth"
            model.save_checkpoint(
                str(final_model_path),
                epoch=optimization_results.get('optimization_rounds', 0),
                loss=optimization_results.get('best_performance', 0.0),
                metrics={'stage': 'feedback_optimized'}
            )
            pipeline_results['final_model_path'] = str(final_model_path)
            
            # Calculate final performance metrics
            final_performance = optimization_results.get('performance_history', [])
            if final_performance:
                best_performance = max(final_performance, key=lambda x: x['dice'])
                pipeline_results['performance_metrics'] = {
                    'best_dice_score': best_performance['dice'],
                    'best_iou_score': best_performance['iou'],
                    'best_accuracy': best_performance['accuracy'],
                    'improvement_over_baseline': optimization_results.get('best_performance', 0.0)
                }
            
            total_time = time.time() - start_time
            pipeline_results['total_time_minutes'] = total_time / 60
            
            logger.info("=" * 80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info(f"Total time: {total_time/60:.2f} minutes")
            logger.info(f"Original data: {len(original_data)} samples")
            logger.info(f"Generated synthetic data: {len(synthetic_data)} samples")
            logger.info(f"Final performance: {pipeline_results['performance_metrics']}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"Pipeline failed at stage: {self.pipeline_state['stage']}")
            logger.error(f"Error: {e}")
            pipeline_results['error'] = str(e)
            pipeline_results['failed_stage'] = self.pipeline_state['stage']
            raise
        
        return pipeline_results
    
    def save_pipeline_summary(self, results: Dict[str, Any], output_path: str):
        """Save a comprehensive pipeline summary"""
        
        summary = {
            'pipeline_configuration': self.config,
            'execution_results': results,
            'system_info': {
                'device': self.device,
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            },
            'recommendations': self._generate_recommendations(results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Pipeline summary saved to {output_path}")
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on pipeline results"""
        recommendations = []
        
        if 'performance_metrics' in results:
            metrics = results['performance_metrics']
            
            if metrics.get('best_dice_score', 0) < 0.7:
                recommendations.append("Consider increasing synthetic data generation or improving generator quality")
            
            if results.get('synthetic_data_count', 0) < results.get('original_data_count', 0) * 3:
                recommendations.append("Generate more synthetic samples per original mask")
            
            if results.get('optimization_results', {}).get('optimization_rounds', 0) < 5:
                recommendations.append("Consider increasing optimization patience for better results")
        
        if not recommendations:
            recommendations.append("Pipeline performed well. Consider fine-tuning for specific use cases.")
        
        return recommendations


def main():
    """Main function for the generative pipeline"""
    parser = argparse.ArgumentParser(description="Generative AI Segmentation Pipeline")
    parser.add_argument("--mode", choices=["setup", "run", "resume"], required=True,
                       help="Pipeline mode")
    parser.add_argument("--images", type=str, required=False,
                       help="Directory containing original images")
    parser.add_argument("--masks", type=str, required=False,
                       help="Directory containing original masks")
    parser.add_argument("--test-images", type=str,
                       help="Directory containing test images (optional)")
    parser.add_argument("--test-masks", type=str,
                       help="Directory containing test masks (optional)")
    parser.add_argument("--config", type=str,
                       help="Configuration file path")
    parser.add_argument("--output-dir", type=str, default="generative_segmentation_project",
                       help="Output directory for the project")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline manager
        pipeline = GenerativePipelineManager(
            base_dir=args.output_dir,
            device=args.device,
            config_path=args.config
        )
        
        if args.mode == "setup":
            # Setup project structure
            config_path = pipeline.setup_project_structure()
            
            print("=" * 80)
            print("GENERATIVE AI SEGMENTATION PIPELINE - SETUP COMPLETE")
            print("=" * 80)
            print()
            print("Project structure created successfully!")
            print()
            print("Next steps:")
            print("1. Add your small dataset:")
            print(f"   - Images: {args.output_dir}/data/original/images/")
            print(f"   - Masks:  {args.output_dir}/data/original/masks/")
            print()
            print("2. Run the complete pipeline:")
            print(f"   python main_generative_pipeline.py --mode run \\")
            print(f"          --images {args.output_dir}/data/original/images \\")
            print(f"          --masks {args.output_dir}/data/original/masks \\")
            print(f"          --config {config_path}")
            print()
            
        elif args.mode == "run":
            # Run complete pipeline
            if not args.images or not args.masks:
                print("Error: --images and --masks are required for run mode")
                return
            
            print("=" * 80)
            print("STARTING GENERATIVE AI SEGMENTATION PIPELINE")
            print("=" * 80)
            
            results = pipeline.run_complete_pipeline(
                images_dir=args.images,
                masks_dir=args.masks,
                test_images_dir=args.test_images,
                test_masks_dir=args.test_masks
            )
            
            # Save comprehensive summary
            summary_path = Path(args.output_dir) / "results" / "pipeline_summary.json"
            pipeline.save_pipeline_summary(results, str(summary_path))
            
            print()
            print("=" * 80)
            print("PIPELINE EXECUTION SUMMARY")
            print("=" * 80)
            print(f"Stages completed: {', '.join(results['stages_completed'])}")
            print(f"Original samples: {results.get('original_data_count', 0)}")
            print(f"Synthetic samples generated: {results.get('synthetic_data_count', 0)}")
            
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                print(f"Best Dice score: {metrics.get('best_dice_score', 0):.4f}")
                print(f"Best IoU score: {metrics.get('best_iou_score', 0):.4f}")
            
            print(f"Total execution time: {results.get('total_time_minutes', 0):.2f} minutes")
            print(f"Final model saved: {results.get('final_model_path', 'N/A')}")
            print(f"Summary saved: {summary_path}")
            print("=" * 80)
            
        elif args.mode == "resume":
            print("Resume functionality not implemented yet")
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"Error: {e}")
        print("Check the log file for detailed error information.")


if __name__ == "__main__":
    print("=" * 80)
    print("GENERATIVE AI SEGMENTATION PIPELINE")
    print("Complete pipeline from small datasets to optimized models")
    print("=" * 80)
    print()
    
    import sys
    if len(sys.argv) == 1:
        print("Usage Examples:")
        print()
        print("1. Setup project structure:")
        print("   python main_generative_pipeline.py --mode setup")
        print()
        print("2. Run complete pipeline:")
        print("   python main_generative_pipeline.py --mode run \\")
        print("          --images data/original/images \\")
        print("          --masks data/original/masks")
        print()
        print("3. Run with test data:")
        print("   python main_generative_pipeline.py --mode run \\")
        print("          --images data/original/images \\")
        print("          --masks data/original/masks \\")
        print("          --test-images data/test/images \\")
        print("          --test-masks data/test/masks")
        print()
        print("For detailed help:")
        print("   python main_generative_pipeline.py --help")
        print()
    else:
        main()