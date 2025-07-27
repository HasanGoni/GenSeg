# UNet Image Segmentation Pipeline

A production-ready, robust implementation of UNet for image segmentation with custom datasets. This codebase is designed by an expert Python programmer with experience in AI/ML systems and follows best practices for maintainability, scalability, and reliability.

## 🌟 Features

- **Production-Ready**: Robust error handling, logging, and validation
- **Easy to Use**: Simple command-line interface and configuration system
- **Flexible**: Supports both binary and multi-class segmentation
- **Complete Pipeline**: From data loading to training to inference
- **Modern Architecture**: Uses PyTorch with best practices
- **Comprehensive Metrics**: Dice coefficient, IoU, pixel accuracy
- **Data Augmentation**: Advanced augmentation pipeline with Albumentations
- **Visualization**: Built-in prediction visualization and overlay generation
- **Checkpointing**: Automatic model saving and resuming
- **Early Stopping**: Prevent overfitting with configurable patience
- **TensorBoard Integration**: Monitor training progress

## 📋 Requirements

- Python 3.7+
- CUDA-capable GPU (recommended) or CPU
- At least 8GB RAM (16GB+ recommended for larger datasets)

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the code
git clone <repository-url>
cd unet-segmentation

# Install dependencies
pip install -r requirements.txt
```

### 2. Setup Project Structure

```bash
python main.py --mode setup
```

This creates the following directory structure:
```
unet_project/
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   ├── val/
│   │   ├── images/
│   │   └── masks/
│   └── test/
│       └── images/
├── checkpoints/
├── logs/
├── predictions/
└── configs/
    └── config.json
```

### 3. Prepare Your Data

Organize your data in the created structure:

**Images**: Place your input images in the respective folders
- Training images: `unet_project/data/train/images/`
- Training masks: `unet_project/data/train/masks/`
- Validation images: `unet_project/data/val/images/`
- Validation masks: `unet_project/data/val/masks/`

**Naming Convention**: 
- Image: `sample001.jpg`
- Corresponding mask: `sample001_mask.jpg` (or configure custom suffix)

**Supported Formats**: JPG, PNG, TIFF

### 4. Configure Training (Optional)

Edit `unet_project/configs/config.json` to customize:

```json
{
  "model": {
    "n_channels": 3,        // RGB=3, Grayscale=1
    "n_classes": 1,         // Binary=1, Multi-class=N
    "bilinear": true
  },
  "training": {
    "batch_size": 4,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "optimizer": "adam",
    "scheduler": "plateau"
  },
  "data": {
    "image_size": [256, 256],
    "mask_suffix": "_mask"
  }
}
```

### 5. Train the Model

```bash
cd unet_project
python ../main.py --mode train --config configs/config.json
```

### 6. Run Inference

**Single Image**:
```bash
python main.py --mode inference --model checkpoints/best_model.pth --input path/to/image.jpg
```

**Batch Processing**:
```bash
python main.py --mode inference --model checkpoints/best_model.pth --input path/to/images/ --output results/
```

## 📚 Detailed Usage

### Configuration Options

#### Model Configuration
- `n_channels`: Input image channels (3 for RGB, 1 for grayscale)
- `n_classes`: Number of output classes (1 for binary, N for multi-class)
- `bilinear`: Use bilinear upsampling vs transposed convolution

#### Training Configuration
- `batch_size`: Training batch size (adjust based on GPU memory)
- `num_epochs`: Maximum training epochs
- `learning_rate`: Initial learning rate
- `weight_decay`: L2 regularization strength
- `optimizer`: Optimizer type ("adam", "adamw", "sgd", "rmsprop")
- `scheduler`: Learning rate scheduler ("plateau", "cosine", "step", "exponential")
- `early_stopping_patience`: Epochs to wait before early stopping

#### Data Configuration
- `image_size`: Target image size [height, width]
- `mask_suffix`: Suffix to identify mask files
- `num_workers`: Number of data loading workers

### Command Line Options

```bash
python main.py --help
```

**Setup Mode**:
```bash
python main.py --mode setup
```

**Training Mode**:
```bash
python main.py --mode train --config path/to/config.json --device cuda
```

**Inference Mode**:
```bash
python main.py --mode inference --model path/to/model.pth --input path/to/input --output path/to/output --device cuda
```

### Programmatic Usage

```python
from unet_model import UNet
from dataset import create_dataloaders
from trainer import UNetTrainer, create_loss_function, create_optimizer
from inference import load_model_for_inference

# Create and train model
model = UNet(n_channels=3, n_classes=1, bilinear=True)
train_loader, val_loader = create_dataloaders(...)
trainer = UNetTrainer(model, train_loader, val_loader, ...)
trainer.train(num_epochs=100)

# Run inference
predictor = load_model_for_inference('checkpoints/best_model.pth')
mask = predictor.predict_single('image.jpg')
predictor.visualize_prediction('image.jpg', save_path='result.png')
```

## 🏗️ Architecture

### Model Architecture
- **Encoder**: 4 downsampling blocks with double convolutions
- **Decoder**: 4 upsampling blocks with skip connections
- **Features**: 64 → 128 → 256 → 512 → 1024 channels
- **Skip Connections**: Preserve spatial information
- **Output**: Configurable for binary or multi-class segmentation

### Key Components

1. **`unet_model.py`**: UNet architecture with robust error handling
2. **`dataset.py`**: Custom dataset class with data validation and augmentation
3. **`trainer.py`**: Comprehensive training pipeline with metrics and checkpointing
4. **`inference.py`**: Production-ready inference with visualization
5. **`main.py`**: Command-line interface for the complete pipeline

### Data Augmentation
- Horizontal/Vertical flips
- Random rotations
- Brightness/Contrast adjustment
- Gaussian noise and blur
- Elastic transformations
- Optical distortions

### Metrics
- **Dice Coefficient**: Overlap measure (F1 score for segmentation)
- **IoU (Intersection over Union)**: Jaccard index
- **Pixel Accuracy**: Percentage of correctly classified pixels

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory**:
- Reduce batch size in config
- Use smaller image size
- Enable gradient checkpointing

**No Data Found**:
- Check file paths and naming convention
- Ensure masks have correct suffix
- Verify supported file formats

**Poor Performance**:
- Increase training epochs
- Adjust learning rate
- Add more training data
- Tune data augmentation

**Import Errors**:
- Install all requirements: `pip install -r requirements.txt`
- Check Python version (3.7+)

### Performance Tips

1. **GPU Usage**: Ensure CUDA is available and properly configured
2. **Data Loading**: Increase `num_workers` if CPU is underutilized
3. **Memory**: Use mixed precision training for larger models
4. **Validation**: Monitor training curves in TensorBoard

## 📊 Monitoring Training

Training progress is logged to TensorBoard:

```bash
tensorboard --logdir unet_project/logs
```

View metrics:
- Training/Validation loss
- Dice coefficient
- IoU score
- Learning rate schedule

## 🎯 Best Practices

### Data Preparation
- Ensure high-quality, consistent annotations
- Balance classes if doing multi-class segmentation
- Use sufficient training data (1000+ samples recommended)
- Validate data integrity before training

### Training
- Start with pre-trained weights if available
- Use learning rate scheduling
- Monitor for overfitting with validation metrics
- Save checkpoints regularly

### Inference
- Preprocess test data consistently with training
- Use appropriate threshold for binary segmentation
- Validate results on known samples
- Consider ensemble methods for critical applications

## 🤝 Contributing

This codebase follows professional software development practices:

- Type hints for better code documentation
- Comprehensive error handling and logging
- Modular design for easy extension
- Configuration-driven approach
- Extensive documentation and examples

## 📄 License

This project is designed for educational and commercial use. Modify and adapt as needed for your specific requirements.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs for error details
3. Ensure your data follows the expected format
4. Verify all dependencies are correctly installed

---

**Built with ❤️ by an Expert Python Programmer**

*This implementation prioritizes code quality, maintainability, and production readiness for real-world applications.*
