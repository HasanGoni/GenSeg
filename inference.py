"""
Inference Pipeline for UNet Model
Handles model loading, prediction, and visualization of results
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Union, Tuple, List, Optional, Dict, Any
from pathlib import Path
import logging

from unet_model import UNet

logger = logging.getLogger(__name__)


class UNetPredictor:
    """
    Production-ready inference pipeline for UNet model
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = 'auto',
        image_size: Tuple[int, int] = (256, 256),
        threshold: float = 0.5
    ):
        """
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('auto', 'cuda', 'cpu')
            image_size: Input image size for the model
            threshold: Threshold for binary segmentation
        """
        self.device = self._get_device(device)
        self.image_size = image_size
        self.threshold = threshold
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Setup preprocessing
        self.preprocess = self._get_preprocessing_pipeline()
        
        logger.info(f"UNet Predictor initialized on {self.device}")
        logger.info(f"Model loaded from: {model_path}")
    
    def _get_device(self, device: str) -> str:
        """Determine the appropriate device for inference"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_path: str) -> UNet:
        """Load trained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model configuration
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                model = UNet(
                    n_channels=config['n_channels'],
                    n_classes=config['n_classes'],
                    bilinear=config['bilinear']
                )
            else:
                # Fallback to default configuration
                logger.warning("Model config not found in checkpoint, using default configuration")
                model = UNet(n_channels=3, n_classes=1, bilinear=True)
            
            # Load model weights
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            
            logger.info(f"Model loaded successfully with {model.get_model_size():,} parameters")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _get_preprocessing_pipeline(self) -> A.Compose:
        """Get preprocessing pipeline for inference"""
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Load and validate input image"""
        if isinstance(image_input, str):
            # Load from file path
            image = cv2.imread(image_input)
            if image is None:
                raise ValueError(f"Could not load image from {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            # Convert PIL Image to numpy array
            image = np.array(image_input.convert('RGB'))
        elif isinstance(image_input, np.ndarray):
            # Use numpy array directly
            image = image_input.copy()
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass  # RGB image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                image = image[:, :, :3]  # Remove alpha channel
            else:
                raise ValueError(f"Unsupported image shape: {image.shape}")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")
        
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input"""
        # Store original size for later use
        self.original_size = image.shape[:2]
        
        # Apply preprocessing
        transformed = self.preprocess(image=image)
        preprocessed_image = transformed['image']
        
        # Add batch dimension
        if len(preprocessed_image.shape) == 3:
            preprocessed_image = preprocessed_image.unsqueeze(0)
        
        return preprocessed_image.to(self.device)
    
    def _postprocess_prediction(self, prediction: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """Postprocess model prediction to original image size"""
        # Remove batch dimension
        if len(prediction.shape) == 4:
            prediction = prediction.squeeze(0)
        
        # Apply sigmoid for binary segmentation
        if self.model.n_classes == 1:
            prediction = torch.sigmoid(prediction).squeeze(0)
        else:
            # For multi-class, take the class with highest probability
            prediction = torch.softmax(prediction, dim=0)
            prediction = prediction[1] if prediction.shape[0] > 1 else prediction[0]
        
        # Convert to numpy and resize to original size
        prediction_np = prediction.cpu().numpy()
        
        if prediction_np.shape != original_size:
            prediction_np = cv2.resize(
                prediction_np, 
                (original_size[1], original_size[0]), 
                interpolation=cv2.INTER_LINEAR
            )
        
        return prediction_np
    
    def predict_single(
        self, 
        image_input: Union[str, np.ndarray, Image.Image],
        return_probability: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict segmentation mask for a single image
        
        Args:
            image_input: Input image (file path, numpy array, or PIL Image)
            return_probability: Whether to return probability map along with binary mask
            
        Returns:
            Binary mask (and probability map if requested)
        """
        # Load and preprocess image
        image = self._load_image(image_input)
        input_tensor = self._preprocess_image(image)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model(input_tensor)
        
        # Postprocess prediction
        prob_map = self._postprocess_prediction(prediction, self.original_size)
        
        # Create binary mask
        binary_mask = (prob_map > self.threshold).astype(np.uint8)
        
        if return_probability:
            return binary_mask, prob_map
        return binary_mask
    
    def predict_batch(
        self, 
        image_list: List[Union[str, np.ndarray, Image.Image]],
        batch_size: int = 4,
        return_probability: bool = False
    ) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
        """
        Predict segmentation masks for a batch of images
        
        Args:
            image_list: List of input images
            batch_size: Batch size for processing
            return_probability: Whether to return probability maps
            
        Returns:
            List of predictions (masks and probability maps if requested)
        """
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch_images = image_list[i:i + batch_size]
            batch_results = []
            
            for image_input in batch_images:
                result = self.predict_single(image_input, return_probability)
                batch_results.append(result)
            
            results.extend(batch_results)
        
        return results
    
    def predict_directory(
        self,
        input_dir: str,
        output_dir: str,
        image_extensions: List[str] = None,
        save_probability_maps: bool = False,
        save_overlays: bool = True
    ):
        """
        Predict segmentation masks for all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save predictions
            image_extensions: Supported image file extensions
            save_probability_maps: Whether to save probability maps
            save_overlays: Whether to save overlay visualizations
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        logger.info(f"Found {len(image_files)} images to process")
        
        for image_file in image_files:
            try:
                # Predict
                if save_probability_maps:
                    mask, prob_map = self.predict_single(str(image_file), return_probability=True)
                else:
                    mask = self.predict_single(str(image_file), return_probability=False)
                
                # Save mask
                mask_filename = f"{image_file.stem}_mask.png"
                mask_path = output_path / mask_filename
                cv2.imwrite(str(mask_path), mask * 255)
                
                # Save probability map if requested
                if save_probability_maps:
                    prob_filename = f"{image_file.stem}_prob.png"
                    prob_path = output_path / prob_filename
                    cv2.imwrite(str(prob_path), (prob_map * 255).astype(np.uint8))
                
                # Save overlay if requested
                if save_overlays:
                    overlay = self.create_overlay(str(image_file), mask)
                    overlay_filename = f"{image_file.stem}_overlay.png"
                    overlay_path = output_path / overlay_filename
                    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                
                logger.info(f"Processed: {image_file.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {image_file.name}: {e}")
    
    def create_overlay(
        self, 
        image_input: Union[str, np.ndarray, Image.Image], 
        mask: np.ndarray,
        alpha: float = 0.6,
        mask_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> np.ndarray:
        """
        Create overlay visualization of image and mask
        
        Args:
            image_input: Original image
            mask: Binary segmentation mask
            alpha: Transparency of the overlay
            mask_color: Color for the mask overlay (RGB)
            
        Returns:
            Overlay image as numpy array
        """
        # Load original image
        image = self._load_image(image_input)
        
        # Resize mask to match original image size if needed
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = mask_color
        
        # Create overlay
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
        
        return overlay
    
    def visualize_prediction(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        save_path: Optional[str] = None,
        show_plot: bool = True,
        figsize: Tuple[int, int] = (15, 5)
    ):
        """
        Visualize prediction results with original image, mask, and overlay
        
        Args:
            image_input: Input image
            save_path: Path to save the visualization
            show_plot: Whether to display the plot
            figsize: Figure size for the plot
        """
        # Get prediction
        mask, prob_map = self.predict_single(image_input, return_probability=True)
        
        # Load original image
        original_image = self._load_image(image_input)
        
        # Create overlay
        overlay = self.create_overlay(image_input, mask)
        
        # Create visualization
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Probability map
        im1 = axes[1].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
        axes[1].set_title('Probability Map')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Binary mask
        axes[2].imshow(mask, cmap='gray')
        axes[2].set_title('Binary Mask')
        axes[2].axis('off')
        
        # Overlay
        axes[3].imshow(overlay)
        axes[3].set_title('Overlay')
        axes[3].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def get_prediction_stats(self, mask: np.ndarray) -> Dict[str, Any]:
        """Get statistics about the prediction"""
        total_pixels = mask.size
        positive_pixels = np.sum(mask > 0)
        
        return {
            'total_pixels': total_pixels,
            'positive_pixels': positive_pixels,
            'negative_pixels': total_pixels - positive_pixels,
            'positive_ratio': positive_pixels / total_pixels,
            'coverage_percentage': (positive_pixels / total_pixels) * 100
        }


def load_model_for_inference(
    model_path: str,
    device: str = 'auto',
    image_size: Tuple[int, int] = (256, 256),
    threshold: float = 0.5
) -> UNetPredictor:
    """
    Convenience function to load a model for inference
    
    Args:
        model_path: Path to the trained model checkpoint
        device: Device to run inference on
        image_size: Input image size for the model
        threshold: Threshold for binary segmentation
        
    Returns:
        UNetPredictor instance ready for inference
    """
    return UNetPredictor(
        model_path=model_path,
        device=device,
        image_size=image_size,
        threshold=threshold
    )


if __name__ == "__main__":
    # Example usage
    print("Inference module loaded successfully!")
    print("Example usage:")
    print("""
    from inference import load_model_for_inference
    
    # Load trained model
    predictor = load_model_for_inference(
        model_path='checkpoints/best_model.pth',
        threshold=0.5
    )
    
    # Predict single image
    mask = predictor.predict_single('path/to/image.jpg')
    
    # Predict with probability map
    mask, prob_map = predictor.predict_single('path/to/image.jpg', return_probability=True)
    
    # Visualize results
    predictor.visualize_prediction('path/to/image.jpg', save_path='result.png')
    
    # Process entire directory
    predictor.predict_directory(
        input_dir='input_images/',
        output_dir='predictions/',
        save_overlays=True
    )
    
    # Get prediction statistics
    stats = predictor.get_prediction_stats(mask)
    print(f"Coverage: {stats['coverage_percentage']:.2f}%")
    """)