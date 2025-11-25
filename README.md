# DyGLNet: Medical Image Segmentation with Global-Local Fusion

Official implementation of the paper **"DyGLNet: Hybrid Global-Local Feature Fusion with Dynamic Upsampling for Medical Image Segmentation"** .

DyGLNet is a lightweight and high-performance network designed for medical image segmentation, addressing key challenges like multi-scale lesions, blurred boundaries, and high computational demands.


## 🔍 Core Features
- **SHDCBlock**: Hybrid feature extractor combining single-head attention and multi-scale dilated convolutions for global-local feature modeling.
- **DyFusionUp**: Dynamic adaptive upsampling module optimizing boundary restoration and small-object segmentation accuracy.
- **Efficiency-Accuracy Balance**: Only 9.98M parameters & 11.16G FLOPs, outperforming state-of-the-art (SOTA) methods on 7 medical datasets.


## 📊 Performance
Achieves SOTA results across multiple tasks:
- Polyp: Kvasir-SEG (91.34% Dice), CVC-ClinicDB (93.71% Dice)
- Brain Tumor: Brain-MRI (88.52% Dice)
- Dermoscopic: ISIC-2016 (91.71% Dice), PH2 (95.40% Dice)
- Pathological: GlaS (94.32% Dice), TNBC (80.05% Dice)


## 📁 Datasets
All datasets used in this research are **publicly available** and can be downloaded online.



## 🙏 Acknowledgments
We sincerely thank the authors of DyT, SHVIT, DySample, and other related open-source projects for their excellent work and publicly available code. Their contributions have provided important technical support and foundations for the completion of this research. 

## 🚀 Quick Start
### Requirements
pip install -r requirements.txt
