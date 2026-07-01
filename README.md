# SF-RTDETR: Lightweight Visual Detection Framework for UAV-Based Wildlife Monitoring

This repository provides the official implementation of the manuscript:

**SF-RTDETR: A Lightweight Real-Time Visual Detection Framework for UAV-Based Wildlife Monitoring**

The proposed framework, named **SF-RTDETR**, is built upon the Ultralytics implementation of RT-DETR. It introduces a lightweight **StarFlex backbone** for real-time UAV-based wildlife monitoring. The StarFlex backbone integrates configurable **FlexConv**, **Multi-Dilation Depthwise Convolution (MDConv)**, and **Efficient Channel Attention (ECA)** to improve multi-scale feature representation while maintaining low computational cost.

---

## Environment Setup

The code was developed and tested under the following environment:

```text
Operating system: Linux
GPU: NVIDIA Tesla V100
Python: 3.10.16
PyTorch: 2.3.0
Ultralytics RT-DETR
Input size: 640 × 640
Batch size: 4
Training epochs: 100
Optimizer: AdamW
Initial learning rate: 1 × 10^-4
Weight decay: 1 × 10^-4
Random seed: 0
Deterministic mode: True
Automatic Mixed Precision: False
Pretrained weights: False
```

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

---

## Repository Structure
```text
model/
├── README.md
├── requirements.txt
├── rtdetr-starflex.yaml
├── starFlex.py
├── train.py
├── val.py
├── get_COCO_metrice.py
├── split_data.py
├── yolo2coco.py
└── samples/
```
README.md: usage instructions, environment settings, dataset preparation, training procedure, validation/testing procedure, COCO-style metric calculation, data availability statement, and reproducibility information.
requirements.txt: Python dependencies required for running the code.
rtdetr-starflex.yaml: model configuration file of SF-RTDETR.
starFlex.py: implementation of the StarFlex backbone and its related modules used by rtdetr-starflex.yaml.
train.py: training script based on the Ultralytics RT-DETR framework.
val.py: validation/testing script used to obtain model performance, parameter number, GFLOPs, FPS, and prediction JSON files.
get_COCO_metrice.py: script used to calculate COCO-style AP metrics based on annotation JSON and prediction JSON files.
split_data.py: auxiliary script used to illustrate the train/validation/test data splitting process.
yolo2coco.py: auxiliary script used to convert YOLO-style annotations into COCO-style JSON files for COCO-style metric calculation.
samples/: a small representative subset of the Tibetan Antelope Dataset used to illustrate the image format, annotation format, and dataset organization.

The file rtdetr-starflex.yaml defines the SF-RTDETR model architecture and uses the modules implemented in starFlex.py.

Additional reproducibility resources may also be included to improve transparency:

```text
model/
├── samples/
│   ├── images/
│   └── labels/
└── data.yaml
```

* `samples/`: a small representative subset of the Tibetan Antelope Dataset used to illustrate the image format, annotation format, and dataset organization.
* `data.yaml`: an example dataset configuration file showing the dataset path and label information.

---

## Dataset Preparation

The models are trained using the YOLO-style annotation format supported by the Ultralytics framework.

The recommended dataset structure is:

```text
datasets/
├── TibetanAntelope/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml
├── BirdsComputerVision/
│   ├── images/
│   ├── labels/
│   └── data.yaml
└── VisDrone2019/
    ├── images/
    ├── labels/
    └── data.yaml
```

The dataset YAML file should follow the standard Ultralytics format, including the training, validation, and test image paths, the number of classes, and the class names.

Although the training scripts use the YOLO-style annotation format required by the Ultralytics framework, all datasets share the same image samples, category definitions, bounding-box annotations, and train/validation/test splits used in the experiments. For COCO-style AP evaluation, the corresponding annotations and prediction results are converted into JSON format and evaluated using `get_COCO_metrice.py`. Therefore, the use of YOLO-style files is only a framework-specific input format and does not change the dataset content or the final COCO-style evaluation protocol.

---

### Tibetan Antelope Dataset

The primary Tibetan Antelope Dataset was constructed for UAV-based wildlife monitoring on the Qinghai--Tibet Plateau. The dataset contains 3000 annotated images after data augmentation, including 2400 training images, 300 validation images, and 300 test images.

The full Tibetan Antelope Dataset is not publicly redistributed in this repository due to data source-use and redistribution restrictions. To support transparency and format verification, we provide a small representative subset in the `samples/` folder, where possible. The sample subset is used only to illustrate the image format, annotation format, and dataset organization.

For academic and non-commercial research purposes, requests for access to the full Tibetan Antelope Dataset can be sent to:

```text
2024388305@stu.zjhu.edu.cn
```

Access will be considered subject to relevant data-use restrictions and reasonable research purposes.

---

### Birds Computer Vision Dataset

The Birds Computer Vision Dataset used in this study is an open-source dataset from Roboflow Universe:

```text
https://universe.roboflow.com/evilsumrak/birds-wnak6
```

Users should download and use this dataset according to the original Roboflow Universe access policy and citation requirements.

---

### VisDrone2019 Dataset

The VisDrone2019 dataset used in this study is available from the official VisDrone repository:

```text
https://github.com/VisDrone/VisDrone-Dataset
```

In this work, we use **Task 1: Object Detection in Images** for evaluation. Users should download and use the dataset according to the official VisDrone data access policy and citation requirements.

---

## Training

Train SF-RTDETR with:

```bash
python train.py
```

Before training, modify the dataset path in `train.py` according to your local environment:

```python
data='path/to/dataset.yaml'
```

The model configuration used for SF-RTDETR is:

```python
model = RTDETR('rtdetr-starflex.yaml')
```

The main training settings used in the paper are:

```text
Input size: 640 × 640
Batch size: 4
Epochs: 100
Optimizer: AdamW
Initial learning rate: 1 × 10^-4
Weight decay: 1 × 10^-4
Random seed: 0
Deterministic mode: True
AMP: False
Pretrained weights: False
```

During training, the best checkpoint is automatically saved by Ultralytics according to validation performance:

```text
runs/train/[experiment_name]/weights/best.pt
```

---

## Validation and Testing

Evaluate the trained model with:

```bash
python val.py
```

Before running the script, modify the following paths in `val.py`:

```python
model_path = 'runs/train/[experiment_name]/weights/best.pt'
data = 'path/to/dataset.yaml'
```

The validation/testing script reports model complexity, inference speed, and detection performance, including:

```text
GFLOPs
Parameters
Preprocessing time
Inference time
Postprocessing time
FPS
Precision
Recall
F1-score
mAP50
mAP75
mAP50:95
```

The results are saved to:

```text
runs/val/[experiment_name]/paper_data.txt
```

When `save_json=True`, the prediction results are also saved as JSON files for COCO-style metric calculation.

---

## COCO-style Metric Calculation

Calculate COCO-style AP metrics with:

```bash
python get_COCO_metrice.py \
  --anno_json path/to/annotation.json \
  --pred_json path/to/predictions.json
```

The script uses `pycocotools` to calculate standard COCO detection metrics and `tidecv` for additional error analysis.

The main COCO-style metrics include:

```text
AP50
AP50:95
AP_S
AP_M
AP_L
```

---

## Reproducing the Main Results

To reproduce the main results reported in the paper:

1. Prepare the dataset in YOLO-style format.
2. Modify the dataset path in `train.py`.
3. Run:

```bash
python train.py
```

4. Modify the checkpoint path and dataset path in `val.py`.
5. Run:

```bash
python val.py
```

6. Use the generated prediction JSON and the corresponding annotation JSON to calculate COCO-style AP metrics:

```bash
python get_COCO_metrice.py \
  --anno_json path/to/annotation.json \
  --pred_json path/to/predictions.json
```

The final results reported in the paper are obtained using the validation-selected `best.pt` checkpoint and evaluating it on the test set.

On the Tibetan Antelope Dataset, SF-RTDETR achieves:

```text
AP50: 93.4%
AP50:95: 64.7%
Params: 12.4M
GFLOPs: 33.6G
FPS: 81.0
```

---

## Checkpoints

The reported results are obtained using the `best.pt` checkpoint selected according to validation performance. This repository encourages full end-to-end reproduction by training the model from scratch using the provided code, configuration file, random seed, and training settings.

If checkpoint-based verification is required for academic purposes, readers may contact the author by email:

```text
2024388305@stu.zjhu.edu.cn
```

---

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

---

## Requirements

The main dependencies include:

```text
ultralytics
torch==2.3.0
torchvision
numpy
opencv-python
pillow
pyyaml
requests
scipy
tqdm
matplotlib
pandas
seaborn
prettytable
pycocotools
tidecv
thop
psutil
py-cpuinfo
```

Please install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code or find this work helpful, please cite the following manuscript:

```bibtex
@article{shi2026sfrtdetr,
  title={SF-RTDETR: A Lightweight Real-Time Visual Detection Framework for UAV-Based Wildlife Monitoring},
  author={Shi, Jian and Huang, Xu and Zeng, Mengjia},
  journal={},
  year={2026},
  note={}
}
```

##
