# StarFlex
# SF-RTDETR: Lightweight Real-Time Detection for UAV Wildlife Surveillance

This repository contains the official implementation of **SF-RTDETR**, a lightweight real-time object detection framework for UAV-based aerial wildlife monitoring. SF-RTDETR is built upon the RT-DETR architecture with a redesigned **StarFlex** backbone, integrating Flexible Convolution (FlexConv), Multi-Dilation Convolution (MDConv), and Efficient Channel Attention (ECA) for efficient multi-scale feature modeling.

---

## Requirements

Python 3.10.16 is recommended. Install dependencies via:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
# Base
matplotlib>=3.3.0
numpy>=1.22.2
opencv-python>=4.6.0
pillow>=7.1.2
pyyaml>=5.3.1
requests>=2.23.0
scipy>=1.4.1
torch>=1.8.0
torchvision>=0.9.0
tqdm>=4.64.0

# Logging
# tensorboard>=2.13.0

# Plotting
pandas>=1.1.4
seaborn>=0.11.0

# Extras
psutil
py-cpuinfo
thop>=0.1.1
```

---

## Training

All experiments were conducted on a Linux system with a Tesla V100 GPU, using PyTorch 2.3.0 and Python 3.10.16. The training settings were kept consistent across all experiments:

| Setting | Value |
|--------|-------|
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Input size | 640 × 640 |
| Batch size | 4 |
| Epochs | 200 |

---

## Model

The StarFlex backbone and the full SF-RTDETR model implementation are provided in the `models/` directory:

```
models/
├── starflex.py       # StarFlex backbone (FlexConv + MDAB)
├── sf_rtdetr.py      # Full SF-RTDETR detection framework
```

---

## Datasets

### 1. Tibetan Antelope Dataset (Primary Wildlife Benchmark)

This is a self-constructed dataset for UAV-based wildlife detection under complex plateau environments. It contains 3,000 images (2,400 train / 300 val / 300 test) annotated using LabelImg, with data augmentation applied via the Albumentations library (rain, shadow, and haze simulation).

The dataset is available upon reasonable request. Please contact the corresponding author at:

📧 **2024388305@stu.zjhu.edu.cn**

### 2. Birds Computer Vision Dataset

A publicly available aerial bird detection dataset from Roboflow, containing 4,539 annotated images under varying illumination conditions (daytime and nighttime).

🔗 **https://universe.roboflow.com/evilsumrak/birds-wnak6**

### 3. VisDrone2019 Dataset

A large-scale UAV benchmark dataset containing 10 object categories across diverse urban aerial scenes, with 6,471 training images, 548 validation images, and 1,610 test images.

🔗 **https://github.com/VisDrone/VisDrone-Dataset**

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{shi2025sfrtdetr,
  title     = {Lightweight Real-Time Detection for UAV Wildlife Surveillance: 
               Handling Scale Variation and Computational Efficiency},
  author    = {Shi, Jian and Huang, Xu and Zeng, Mengjia},
  journal   = {Pattern Analysis and Applications},
  year      = {2025}
}
```

---

## License

This project is released for academic research purposes only.
