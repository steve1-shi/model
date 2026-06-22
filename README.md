# SF-RTDETR: Lightweight Visual Detection Framework for UAV-Based Wildlife Monitoring

This repository provides the implementation details of the manuscript:

**Lightweight Visual Detection Framework for UAV-Based Wildlife Monitoring on Qinghai--Tibet Plateau**

The proposed framework, named **SF-RTDETR**, is built upon RT-DETR and introduces a lightweight **StarFlex backbone** for real-time UAV-based wildlife monitoring. The backbone integrates configurable **FlexConv**, **Multi-Dilation Depthwise Convolution (MDConv)**, and **Efficient Channel Attention (ECA)** to improve multi-scale feature representation while maintaining low computational cost.

## Repository Structure

```text
model/
├── README.md
├── requirements.txt
├── rtdetr-starflex.yaml
└── starFlex.py
```

* `starFlex.py`: implementation of the proposed StarFlex backbone and related modules.
* `rtdetr-starflex.yaml`: model configuration file used for SF-RTDETR.
* `requirements.txt`: Python dependencies required for running the implementation.
* `README.md`: usage instructions and experimental settings.

## Environment Requirements

The experiments in the manuscript were conducted under the following environment:

```text
Operating system: Linux
GPU: NVIDIA Tesla V100
Python: 3.10.16
PyTorch: 2.3.0
Input size: 640 × 640
Batch size: 4
Training epochs: 100
Optimizer: AdamW
Initial learning rate: 1 × 10^-4
Weight decay: 1 × 10^-4
```

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

All datasets used in this study are converted into COCO-style annotation format.

The recommended dataset structure is:

```text
datasets/
├── TibetanAntelope/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── annotations/
│       ├── train.json
│       ├── val.json
│       └── test.json
├── BirdsComputerVision/
│   ├── images/
│   └── annotations/
└── VisDrone2019/
    ├── images/
    └── annotations/
```

The primary Tibetan Antelope Dataset was constructed for UAV-based wildlife monitoring on the Qinghai--Tibet Plateau. The dataset contains 3000 annotated images after data augmentation, including 2400 training images, 300 validation images, and 300 test images.

The additional evaluation datasets include:

* Birds Computer Vision Dataset
* VisDrone2019 Dataset

The Tibetan Antelope Dataset is available from the authors upon reasonable request. The Birds Computer Vision and VisDrone2019 datasets follow their original data access policies.

## Training

To train SF-RTDETR, place the dataset in the required directory and use the provided configuration file.

Example training command:

```bash
python tools/train.py -c rtdetr-starflex.yaml
```

If your RT-DETR implementation uses a different project structure, copy `starFlex.py` and `rtdetr-starflex.yaml` into the corresponding model and configuration directories, and then run the training script following your local RT-DETR framework.

The training settings used in the manuscript are:

```text
Input size: 640 × 640
Batch size: 4
Epochs: 100
Optimizer: AdamW
Initial learning rate: 1 × 10^-4
Weight decay: 1 × 10^-4
```

## Evaluation

Example evaluation command:

```bash
python tools/eval.py -c rtdetr-starflex.yaml -r path/to/checkpoint.pth
```

The model is evaluated using the standard COCO evaluation protocol. The main metrics include:

```text
AP50
AP50:95
AP_S
AP_M
AP_L
Params
GFLOPs
FPS
```

## Main Results

On the Tibetan Antelope Dataset, SF-RTDETR achieves:

```text
AP50: 93.4%
AP50:95: 64.7%
Params: 12.4M
GFLOPs: 33.6G
FPS: 81.0
```

Compared with RT-DETR-R18, SF-RTDETR reduces computational complexity from 56.9 GFLOPs to 33.6 GFLOPs while maintaining competitive detection accuracy and improving inference speed.

## Key Modules

### FlexConv

FlexConv is implemented as a configurable convolutional wrapper. By adjusting the kernel size and group number, it can represent different convolution behaviors. In this work, depthwise convolution and pointwise convolution are used as two representative configurations:

```text
Depthwise Conv: k > 1, g = C_in
Pointwise Conv: k = 1, g = 1
```

Depthwise convolution is used for efficient spatial feature extraction, while pointwise convolution is used for channel interaction and feature fusion.

### MDConv

MDConv adopts parallel depthwise convolution branches with different dilation rates to enlarge the effective receptive field and enhance multi-scale spatial feature modeling.

### ECA

Efficient Channel Attention is used to recalibrate channel-wise responses with low computational overhead.

## Citation

If you use this code or find this work helpful, please cite the following manuscript:

```bibtex
@article{shi2026sfrtdetr,
  title={Lightweight Visual Detection Framework for UAV-Based Wildlife Monitoring on Qinghai--Tibet Plateau},
  author={Shi, Jian and Huang, Xu and Zeng, Mengjia},
  journal={},
  year={2026},
  note={Manuscript submitted}
}
```

## Note

This repository is directly related to the manuscript submitted to *The Visual Computer*. Please cite the corresponding manuscript if you use this code, configuration, or implementation details in your research.
