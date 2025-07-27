"""
Feedback Optimization System for Generative AI Segmentation Pipeline
Uses segmentation performance to improve synthetic data generation quality

Author: Expert Python Programmer
Purpose: Optimize synthetic data generation based on segmentation feedback
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_score
from collections import deque
import random

from unet_model import UNet
from synthetic_data_generator import SyntheticDataGenerator
from trainer import SegmentationMetrics
from inference import UNetPredictor

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """Track segmentation performance over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.dice_scores = deque(maxlen=window_size)
        self.iou_scores = deque(maxlen=window_size)
        self.pixel_accuracies = deque(maxlen=window_size)
        self.loss_values = deque(maxlen=window_size)
        
        self.best_dice = 0.0
        self.best_iou = 0.0
        self.improvement_threshold = 0.01
        
    def update(self, dice: float, iou: float, accuracy: float, loss: float):
        """Update performance metrics"""
        self.dice_scores.append(dice)
        self.iou_scores.append(iou)
        self.pixel_accuracies.append(accuracy)
        self.loss_values.append(loss)
        
        # Update best scores
        if dice > self.best_dice:
            self.best_dice = dice
        if iou > self.best_iou:
            self.best_iou = iou
    
    def get_current_performance(self) -> Dict[str, float]:
        """Get current average performance"""
        if not self.dice_scores:
            return {'dice': 0.0, 'iou': 0.0, 'accuracy': 0.0, 'loss': float('inf')}
        
        return {
            'dice': np.mean(list(self.dice_scores)),
            'iou': np.mean(list(self.iou_scores)),
            'accuracy': np.mean(list(self.pixel_accuracies)),
            'loss': np.mean(list(self.loss_values))
        }
    
    def is_improving(self) -> bool:
        """Check if performance is improving"""
        if len(self.dice_scores) < self.window_size // 2:
            return True  # Continue optimizing in early stages
        
        current_dice = np.mean(list(self.dice_scores)[-10:])
        previous_dice = np.mean(list(self.dice_scores)[-20:-10])
        
        return current_dice > previous_dice + self.improvement_threshold
    
    def get_improvement_rate(self) -> float:
        """Calculate rate of improvement"""
        if len(self.dice_scores) < 20:
            return 0.0
        
        recent = np.mean(list(self.dice_scores)[-10:])
        older = np.mean(list(self.dice_scores)[-20:-10])
        
        return (recent - older) / older if older > 0 else 0.0


class SyntheticDataQualityAssessor:
    """Assess quality of synthetic data for segmentation"""
    
    def __init__(self, reference_metrics: Dict[str, float]):
        self.reference_metrics = reference_metrics
        self.quality_threshold = 0.8  # Minimum quality score
        
    def assess_sample_quality(
        self,
        synthetic_image: np.ndarray,
        synthetic_mask: np.ndarray,
        real_image: np.ndarray,
        real_mask: np.ndarray
    ) -> Dict[str, float]:
        """Assess quality of a synthetic sample"""
        
        # 1. Structural similarity
        structural_score = self._calculate_structural_similarity(
            synthetic_image, real_image
        )
        
        # 2. Mask consistency
        mask_score = self._calculate_mask_consistency(
            synthetic_mask, real_mask
        )
        
        # 3. Feature preservation
        feature_score = self._calculate_feature_preservation(
            synthetic_image, real_image
        )
        
        # 4. Diversity score
        diversity_score = self._calculate_diversity_score(
            synthetic_image, real_image
        )
        
        # Combined quality score
        quality_score = (
            0.3 * structural_score +
            0.3 * mask_score +
            0.2 * feature_score +
            0.2 * diversity_score
        )
        
        return {
            'structural_similarity': structural_score,
            'mask_consistency': mask_score,
            'feature_preservation': feature_score,
            'diversity': diversity_score,
            'overall_quality': quality_score,
            'is_high_quality': quality_score >= self.quality_threshold
        }
    
    def _calculate_structural_similarity(
        self,
        synthetic: np.ndarray,
        real: np.ndarray
    ) -> float:
        """Calculate structural similarity between images"""
        # Convert to grayscale if needed
        if len(synthetic.shape) == 3:
            synthetic_gray = cv2.cvtColor(synthetic, cv2.COLOR_RGB2GRAY)
            real_gray = cv2.cvtColor(real, cv2.COLOR_RGB2GRAY)
        else:
            synthetic_gray = synthetic
            real_gray = real
        
        # Calculate SSIM-like metric
        mean_syn = np.mean(synthetic_gray)
        mean_real = np.mean(real_gray)
        
        std_syn = np.std(synthetic_gray)
        std_real = np.std(real_gray)
        
        covariance = np.mean((synthetic_gray - mean_syn) * (real_gray - mean_real))
        
        # Simplified SSIM formula
        c1, c2 = 0.01, 0.03
        numerator = (2 * mean_syn * mean_real + c1) * (2 * covariance + c2)
        denominator = (mean_syn**2 + mean_real**2 + c1) * (std_syn**2 + std_real**2 + c2)
        
        return numerator / (denominator + 1e-8)
    
    def _calculate_mask_consistency(
        self,
        synthetic_mask: np.ndarray,
        real_mask: np.ndarray
    ) -> float:
        """Calculate consistency between masks"""
        # Normalize masks
        syn_norm = (synthetic_mask > 128).astype(np.uint8)
        real_norm = (real_mask > 128).astype(np.uint8)
        
        # Calculate IoU
        intersection = np.logical_and(syn_norm, real_norm).sum()
        union = np.logical_or(syn_norm, real_norm).sum()
        
        return intersection / (union + 1e-8)
    
    def _calculate_feature_preservation(
        self,
        synthetic: np.ndarray,
        real: np.ndarray
    ) -> float:
        """Calculate how well features are preserved"""
        # Calculate edge preservation
        syn_edges = cv2.Canny(
            cv2.cvtColor(synthetic, cv2.COLOR_RGB2GRAY) if len(synthetic.shape) == 3 else synthetic,
            50, 150
        )
        real_edges = cv2.Canny(
            cv2.cvtColor(real, cv2.COLOR_RGB2GRAY) if len(real.shape) == 3 else real,
            50, 150
        )
        
        # Edge similarity
        edge_similarity = np.sum(syn_edges & real_edges) / (np.sum(syn_edges | real_edges) + 1e-8)
        
        return edge_similarity
    
    def _calculate_diversity_score(
        self,
        synthetic: np.ndarray,
        real: np.ndarray
    ) -> float:
        """Calculate diversity score (how different but still realistic)"""
        # Calculate histogram differences
        syn_hist = cv2.calcHist([synthetic], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        real_hist = cv2.calcHist([real], [0, 1, 2], None, [32, 32, 32], [0, 256, 0, 256, 0, 256])
        
        # Normalize histograms
        syn_hist = syn_hist / (syn_hist.sum() + 1e-8)
        real_hist = real_hist / (real_hist.sum() + 1e-8)
        
        # Calculate histogram distance (lower = more similar)
        hist_distance = cv2.compareHist(syn_hist, real_hist, cv2.HISTCMP_BHATTACHARYYA)
        
        # Convert to diversity score (0.2-0.8 range is good diversity)
        diversity = np.clip(hist_distance, 0.2, 0.8)
        return (diversity - 0.2) / 0.6  # Normalize to [0, 1]


class AdaptiveGeneratorOptimizer:
    """Optimize generator parameters based on feedback"""
    
    def __init__(
        self,
        generator: SyntheticDataGenerator,
        initial_lr: float = 0.0001,
        adaptation_rate: float = 0.1
    ):
        self.generator = generator
        self.initial_lr = initial_lr
        self.adaptation_rate = adaptation_rate
        
        # Optimization parameters
        self.current_lr = initial_lr
        self.noise_scale = 0.1
        self.diversity_weight = 1.0
        self.quality_weight = 1.0
        
        # Performance history
        self.optimization_history = []
        
    def adapt_parameters(
        self,
        performance_feedback: Dict[str, float],
        quality_feedback: Dict[str, float]
    ):
        """Adapt generator parameters based on feedback"""
        
        current_performance = performance_feedback['dice']
        quality_score = quality_feedback['overall_quality']
        
        # Adjust learning rate based on performance
        if current_performance > 0.8:
            self.current_lr *= 0.95  # Reduce LR when performing well
        elif current_performance < 0.6:
            self.current_lr *= 1.05  # Increase LR when performing poorly
        
        # Adjust noise scale for diversity
        if quality_feedback['diversity'] < 0.3:
            self.noise_scale *= 1.1  # Increase diversity
        elif quality_feedback['diversity'] > 0.8:
            self.noise_scale *= 0.9  # Reduce diversity
        
        # Clip values to reasonable ranges
        self.current_lr = np.clip(self.current_lr, 1e-6, 1e-2)
        self.noise_scale = np.clip(self.noise_scale, 0.01, 0.5)
        
        # Record optimization step
        self.optimization_history.append({
            'performance': current_performance,
            'quality': quality_score,
            'learning_rate': self.current_lr,
            'noise_scale': self.noise_scale
        })
        
        logger.info(f"Adapted parameters - LR: {self.current_lr:.6f}, "
                   f"Noise: {self.noise_scale:.3f}")
    
    def get_optimized_parameters(self) -> Dict[str, float]:
        """Get current optimized parameters"""
        return {
            'learning_rate': self.current_lr,
            'noise_scale': self.noise_scale,
            'diversity_weight': self.diversity_weight,
            'quality_weight': self.quality_weight
        }


class FeedbackOptimizer:
    """
    Main feedback optimization system
    """
    
    def __init__(
        self,
        segmentation_model: UNet,
        synthetic_generator: SyntheticDataGenerator,
        device: str = 'cuda',
        optimization_patience: int = 10
    ):
        self.segmentation_model = segmentation_model
        self.synthetic_generator = synthetic_generator
        self.device = device
        self.optimization_patience = optimization_patience
        
        # Initialize components
        self.performance_tracker = PerformanceTracker()
        self.quality_assessor = None  # Will be initialized with reference data
        self.parameter_optimizer = AdaptiveGeneratorOptimizer(synthetic_generator)
        
        # Optimization state
        self.optimization_rounds = 0
        self.stagnation_counter = 0
        self.best_synthetic_data = []
        
        logger.info("Feedback optimizer initialized")
    
    def initialize_reference_metrics(self, reference_data: List[Dict[str, np.ndarray]]):
        """Initialize quality assessor with reference data"""
        # Calculate reference metrics from real data
        reference_metrics = self._calculate_reference_metrics(reference_data)
        self.quality_assessor = SyntheticDataQualityAssessor(reference_metrics)
        
        logger.info("Reference metrics initialized")
    
    def _calculate_reference_metrics(
        self,
        reference_data: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Calculate baseline metrics from reference data"""
        # This would calculate various quality metrics from real data
        # For now, return reasonable defaults
        return {
            'mean_structural_similarity': 0.8,
            'mean_mask_consistency': 0.9,
            'mean_feature_preservation': 0.85,
            'mean_diversity': 0.6
        }
    
    def evaluate_segmentation_performance(
        self,
        test_images: List[np.ndarray],
        test_masks: List[np.ndarray]
    ) -> Dict[str, float]:
        """Evaluate current segmentation model performance"""
        
        predictor = UNetPredictor(
            model_path=None,  # Use current model in memory
            device=self.device
        )
        predictor.model = self.segmentation_model
        
        total_dice = 0.0
        total_iou = 0.0
        total_accuracy = 0.0
        total_loss = 0.0
        
        metrics_calculator = SegmentationMetrics()
        
        for image, mask in zip(test_images, test_masks):
            # Predict
            pred_mask = predictor.predict_single(image, return_probability=False)
            
            # Convert to tensors for metric calculation
            pred_tensor = torch.from_numpy(pred_mask.astype(np.float32))
            true_tensor = torch.from_numpy(mask.astype(np.float32))
            
            # Calculate metrics
            dice = metrics_calculator.dice_coefficient(pred_tensor, true_tensor)
            iou = metrics_calculator.iou_score(pred_tensor, true_tensor)
            accuracy = metrics_calculator.pixel_accuracy(pred_tensor, true_tensor)
            
            total_dice += dice
            total_iou += iou
            total_accuracy += accuracy
        
        num_samples = len(test_images)
        performance = {
            'dice': total_dice / num_samples,
            'iou': total_iou / num_samples,
            'accuracy': total_accuracy / num_samples,
            'loss': total_loss / num_samples
        }
        
        # Update performance tracker
        self.performance_tracker.update(**performance)
        
        return performance
    
    def assess_synthetic_data_quality(
        self,
        synthetic_data: List[Dict[str, np.ndarray]],
        reference_data: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Assess quality of generated synthetic data"""
        
        if not self.quality_assessor:
            self.initialize_reference_metrics(reference_data)
        
        quality_scores = []
        
        for syn_sample in synthetic_data:
            # Select random reference sample for comparison
            ref_sample = random.choice(reference_data)
            
            quality = self.quality_assessor.assess_sample_quality(
                syn_sample['image'],
                syn_sample['mask'],
                ref_sample['image'],
                ref_sample['mask']
            )
            quality_scores.append(quality)
        
        # Aggregate quality scores
        avg_quality = {
            'structural_similarity': np.mean([q['structural_similarity'] for q in quality_scores]),
            'mask_consistency': np.mean([q['mask_consistency'] for q in quality_scores]),
            'feature_preservation': np.mean([q['feature_preservation'] for q in quality_scores]),
            'diversity': np.mean([q['diversity'] for q in quality_scores]),
            'overall_quality': np.mean([q['overall_quality'] for q in quality_scores]),
            'high_quality_ratio': np.mean([q['is_high_quality'] for q in quality_scores])
        }
        
        return avg_quality
    
    def optimize_synthetic_generation(
        self,
        real_data: List[Dict[str, np.ndarray]],
        test_data: List[Dict[str, np.ndarray]],
        max_optimization_rounds: int = 20,
        synthetic_samples_per_round: int = 50
    ) -> Dict[str, Any]:
        """
        Main optimization loop
        """
        
        logger.info(f"Starting feedback optimization for {max_optimization_rounds} rounds")
        
        optimization_results = {
            'performance_history': [],
            'quality_history': [],
            'parameter_history': [],
            'best_performance': 0.0,
            'final_synthetic_data': []
        }
        
        # Initialize reference metrics
        self.initialize_reference_metrics(real_data)
        
        for round_num in range(max_optimization_rounds):
            logger.info(f"Optimization round {round_num + 1}/{max_optimization_rounds}")
            
            # 1. Generate synthetic data with current parameters
            masks = [torch.from_numpy(sample['mask']).float() for sample in real_data]
            synthetic_data = self.synthetic_generator.generate_synthetic_data(
                input_masks=masks,
                num_variations=synthetic_samples_per_round // len(masks),
                output_dir=f'synthetic_data_round_{round_num}'
            )
            
            # 2. Evaluate segmentation performance on test data
            test_images = [sample['image'] for sample in test_data]
            test_masks = [sample['mask'] for sample in test_data]
            performance = self.evaluate_segmentation_performance(test_images, test_masks)
            
            # 3. Assess synthetic data quality
            quality = self.assess_synthetic_data_quality(synthetic_data, real_data)
            
            # 4. Check for improvement
            current_performance = performance['dice']
            is_improving = self.performance_tracker.is_improving()
            
            if current_performance > optimization_results['best_performance']:
                optimization_results['best_performance'] = current_performance
                optimization_results['final_synthetic_data'] = synthetic_data
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # 5. Adapt generator parameters
            self.parameter_optimizer.adapt_parameters(performance, quality)
            
            # 6. Record results
            optimization_results['performance_history'].append(performance)
            optimization_results['quality_history'].append(quality)
            optimization_results['parameter_history'].append(
                self.parameter_optimizer.get_optimized_parameters()
            )
            
            # 7. Log progress
            logger.info(f"Round {round_num + 1} - Performance: {current_performance:.4f}, "
                       f"Quality: {quality['overall_quality']:.4f}")
            
            # 8. Check early stopping
            if self.stagnation_counter >= self.optimization_patience:
                logger.info(f"Early stopping after {round_num + 1} rounds due to stagnation")
                break
            
            if not is_improving and round_num > 5:
                logger.info(f"Stopping optimization - no improvement detected")
                break
        
        logger.info(f"Optimization completed. Best performance: "
                   f"{optimization_results['best_performance']:.4f}")
        
        return optimization_results
    
    def visualize_optimization_progress(
        self,
        optimization_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """Visualize optimization progress"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        rounds = range(1, len(optimization_results['performance_history']) + 1)
        
        # Performance metrics
        performance_history = optimization_results['performance_history']
        axes[0, 0].plot(rounds, [p['dice'] for p in performance_history], 'b-', label='Dice')
        axes[0, 0].plot(rounds, [p['iou'] for p in performance_history], 'r-', label='IoU')
        axes[0, 0].set_title('Segmentation Performance')
        axes[0, 0].set_xlabel('Optimization Round')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Quality metrics
        quality_history = optimization_results['quality_history']
        axes[0, 1].plot(rounds, [q['overall_quality'] for q in quality_history], 'g-', label='Overall Quality')
        axes[0, 1].plot(rounds, [q['diversity'] for q in quality_history], 'm-', label='Diversity')
        axes[0, 1].set_title('Synthetic Data Quality')
        axes[0, 1].set_xlabel('Optimization Round')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Parameter evolution
        param_history = optimization_results['parameter_history']
        axes[1, 0].plot(rounds, [p['learning_rate'] for p in param_history], 'c-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Evolution')
        axes[1, 0].set_xlabel('Optimization Round')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True)
        
        # Noise scale evolution
        axes[1, 1].plot(rounds, [p['noise_scale'] for p in param_history], 'orange', label='Noise Scale')
        axes[1, 1].set_title('Noise Scale Evolution')
        axes[1, 1].set_xlabel('Optimization Round')
        axes[1, 1].set_ylabel('Noise Scale')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization visualization saved to {save_path}")
        
        plt.show()
    
    def save_optimization_results(
        self,
        optimization_results: Dict[str, Any],
        save_path: str
    ):
        """Save optimization results to file"""
        
        # Prepare data for JSON serialization
        serializable_results = {
            'performance_history': optimization_results['performance_history'],
            'quality_history': optimization_results['quality_history'],
            'parameter_history': optimization_results['parameter_history'],
            'best_performance': optimization_results['best_performance'],
            'optimization_rounds': len(optimization_results['performance_history']),
            'final_synthetic_data_count': len(optimization_results['final_synthetic_data'])
        }
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Optimization results saved to {save_path}")


if __name__ == "__main__":
    print("Feedback Optimizer loaded successfully!")
    print("Example usage:")
    print("""
    from feedback_optimizer import FeedbackOptimizer
    from unet_model import UNet
    from synthetic_data_generator import SyntheticDataGenerator
    
    # Initialize components
    model = UNet(n_channels=3, n_classes=1)
    generator = SyntheticDataGenerator(model_type='pix2pix')
    
    # Create optimizer
    optimizer = FeedbackOptimizer(model, generator)
    
    # Run optimization
    results = optimizer.optimize_synthetic_generation(
        real_data=real_samples,
        test_data=test_samples,
        max_optimization_rounds=20
    )
    
    # Visualize results
    optimizer.visualize_optimization_progress(results, 'optimization_results.png')
    """)