# GenSeg Framework Architecture

This diagram illustrates how the GenSeg (Generative AI for Medical Image Segmentation) framework works in ultra low-data regimes.

```mermaid
graph TD
    %% Data Sources
    A[JSRT Dataset<br/>In-Domain<br/>Train/Val: ~9 labeled samples] --> B[Data Preprocessing<br/>util/JSRT_loader.py]
    A1[NLM MC Dataset<br/>Out-of-Domain<br/>Test] --> B1[Test Data Loading]
    A2[NLM SZ Dataset<br/>Out-of-Domain<br/>Test] --> B1
    
    %% Stage 1: Pix2Pix Pre-training
    B --> C[Stage 1: Pix2Pix Pre-training<br/>scripts/train_pix2pix_lung.sh]
    C --> D[Pix2Pix GAN Model<br/>models_pix2pix/pix2pix_model.py]
    D --> E[Generator G<br/>Mask → Image]
    D --> F[Discriminator D<br/>Real vs Fake Images]
    
    %% Stage 2: End-to-End Training with Betty Framework
    E --> G[Stage 2: End-to-End Training<br/>scripts/train_end2end_jsrt.sh]
    G --> H[Betty Bilevel Optimization<br/>betty.engine]
    
    %% Upper Level Problem (Architecture/Generator)
    H --> I[Upper Level: Architecture Problem<br/>Arch Class]
    I --> J[Optimize Generator Parameters<br/>Based on Segmentation Performance]
    
    %% Lower Level Problem (Segmentation)
    H --> K[Lower Level: Segmentation Problem<br/>Segm Class]
    K --> L[U-Net Segmentation Model<br/>unet/unet_model.py]
    
    %% Training Loop Components
    L --> M[Real Image + Real Mask Loss<br/>Cross-entropy + Dice Loss]
    E --> N[Generated Fake Images<br/>From Real Masks]
    N --> O[Fake Image + Real Mask Loss<br/>Weighted by λ]
    
    M --> P[Combined Loss Function<br/>L_real + λ × L_fake]
    O --> P
    P --> K
    
    %% Validation and Testing
    B1 --> Q[Testing Phase<br/>scripts/test_lung.sh]
    L --> Q
    Q --> R[Evaluation Metrics<br/>Dice Score, IoU]
    
    %% Alternative Generative Models
    S[BBDM Extension<br/>Diffusion-based] --> T[Alternative Generators]
    S1[Soft-intro VAE<br/>Extension] --> T
    S2[WGAN-GP<br/>Simultaneous Generation] --> T
    T --> G
    
    %% 3D Extension
    U[GenSeg-3D<br/>3D Medical Segmentation] --> V[3D U-Net<br/>UNet3D/]
    V --> W[3D Training Pipeline<br/>train_end2end.py]
    
    %% Monitoring and Logging
    G --> X[WandB Logging<br/>Training Metrics]
    Q --> Y[Performance Evaluation<br/>In-domain + Out-of-domain]
    
    %% Model Checkpoints
    L --> Z[Model Checkpoints<br/>checkpoint_lung_new/]
    Z --> AA[Pre-trained Models<br/>Pix2Pix + U-Net]
    
    %% Styling
    classDef dataBox fill:#e1f5fe
    classDef modelBox fill:#f3e5f5
    classDef processBox fill:#e8f5e8
    classDef extensionBox fill:#fff3e0
    
    class A,A1,A2,B,B1 dataBox
    class D,E,F,L,S,S1,S2,U,V modelBox
    class C,G,H,I,K,Q,W processBox
    class T,AA extensionBox
```

## Key Components Explained:

### 1. **Two-Stage Training Process**
- **Stage 1**: Pre-train Pix2Pix GAN on limited labeled data (mask→image generation)
- **Stage 2**: End-to-end training using Betty bilevel optimization framework

### 2. **Bilevel Optimization (Betty Framework)**
- **Upper Level**: Optimizes generator parameters based on segmentation performance
- **Lower Level**: Trains segmentation model using both real and generated data
- **Key Innovation**: Generator learns to create images that improve segmentation

### 3. **Data Augmentation Strategy**
- Uses real masks to generate synthetic images via trained Pix2Pix
- Combines real image-mask pairs with synthetic image-real mask pairs
- Loss weighting parameter λ controls synthetic data contribution

### 4. **Architecture Flexibility**
- **Segmentation Models**: U-Net, DeepLab, DeepLabV2, Swin-UNet
- **Generative Models**: Pix2Pix (default), BBDM (diffusion), Soft-intro VAE
- **Extensions**: 3D segmentation, simultaneous image-mask generation

### 5. **Evaluation Strategy**
- **In-domain**: JSRT dataset (lung segmentation)
- **Out-of-domain**: NLM MC and NLM SZ datasets
- **Metrics**: Dice Score, IoU
- **Ultra low-data**: Trained with only 9 labeled samples

This framework enables effective medical image segmentation even with extremely limited labeled data by leveraging generative AI for intelligent data augmentation.