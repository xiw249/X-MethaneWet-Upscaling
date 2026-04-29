# X-MethaneWet-Upscaling

A patch-augmented HybridCNNLSTM pipeline for wetland methane (CH4) emission upscaling using the X-MethaneWet dataset, TEM-MDM simulation data, and FLUXNET-CH4 observations.

This repository extends the original X-MethaneWet workflow by adding local 3×3 spatial patch inputs alongside the original point-based temporal features. The goal is to reduce the point-to-area mismatch in methane emission upscaling and evaluate whether simulation pretraining on TEM-MDM can improve FLUXNET-CH4 finetuning.

## Overview

X-MethaneWet is a cross-scale global wetland methane benchmark dataset that combines physics-based TEM-MDM simulation data and real-world FLUXNET-CH4 tower observations. The original pipeline represents each sample as a point-based yearly sequence with shape `(365, 15)`.

In this project, we extend that representation to include both:

- Point input: `(365, 15)`
- Local spatial patch input: `(365, 15, 3, 3)`

The patch input is centered on the same grid cell as the original point input and provides local neighborhood context. We implement a HybridCNNLSTM model that combines a CNN-based spatial branch with an LSTM-based temporal branch.

## Project Context

This repository was developed to address the specific task of wetland methane emission upscaling. While exploring data mining and feature attribution approaches for the existing pipeline, we identified a fundamental bottleneck: a point-to-area spatial mismatch.

To tackle this upscaling challenge, we focused on improving the input representation. This repository contains the data engineering and architectural modifications implemented to support a patch-augmented HybridCNNLSTM pipeline, providing a stronger foundation for grid-level methane emission prediction.

## Main Contributions

- Added patch-augmented preprocessing for FLUXNET-CH4.
- Added patch-augmented preprocessing for TEM-MDM simulation data.
- Implemented `HybridCNNLSTM` in `model.py`.
- Modified `pretrain.py` to support dual-input simulation pretraining.
- Modified `finetune.py` to support dual-input FLUXNET finetuning.
- Added best-checkpoint saving for TEM-MDM pretraining.
- Evaluated scratch finetuning, TEM-MDM pretraining, and pretrained FLUXNET finetuning.

## Repository Structure

```text
.
├── code/
│   ├── data_processing/
│   │   ├── FLUXNET-CH4.py      # Generates point and patch inputs for FLUXNET-CH4
│   │   └── TEM-MDM.py          # Generates point and patch inputs for TEM-MDM
│   │
│   ├── model_training/
│   │   ├── adversarial.py      # Original adversarial transfer learning script
│   │   ├── base_model.py       # Original base model training script
│   │   ├── config.py           # Configuration settings
│   │   ├── finetune.py         # Scratch and pretrained FLUXNET finetuning
│   │   ├── model.py            # Model architectures, including HybridCNNLSTM
│   │   ├── pretrain.py         # TEM-MDM pretraining with best-checkpoint saving
│   │   ├── residual.py         # Original residual learning script
│   │   └── reweight.py         # Original reweighting script
│   │
│   └── model_save/
│       └── hybrid_cnn_lstm/
│           └── base_model.pth  # Small pretrained HybridCNNLSTM checkpoint
│
├── requirements.txt
└── README.md
```

## Data

Raw X-MethaneWet data are not included in this repository because the dataset files are large. Please follow the original X-MethaneWet repository instructions to download the data, then place the data under:

```text
data/
```

Original X-MethaneWet repository:

```text
https://github.com/ymsun99/X-MethaneWet
```

The expected high-level data structure is:

```text
data/
├── TEM-MDM/
└── FLUXNET-CH4/
```

After preprocessing, generated files should be saved under:

```text
processed_data/
├── TEM-MDM/
│   ├── temporal/
│   └── spatial/
└── FLUXNET-CH4/
    ├── temporal/
    └── spatial/
```

Important generated files include:

```text
# FLUXNET temporal
train_data_x.npy
train_patch_x.npy
train_data_y.npy
test_data_x.npy
test_patch_x.npy
test_data_y.npy

# TEM-MDM temporal
input_YYYY.npy
patch_input_YYYY.npy
output_YYYY.npy
```

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The provided `requirements.txt` contains the core dependencies needed for the HybridCNNLSTM preprocessing, pretraining, and finetuning pipeline. The original transformer-based models may require additional dependencies from the Time-Series-Library repository.

If using the original transformer-based models, clone the Time Series Library into the `model_training/` directory:

```bash
cd code/model_training
git clone https://github.com/thuml/Time-Series-Library.git
```

## Preprocessing

Run the preprocessing scripts first:

```bash
cd code/data_processing

python FLUXNET-CH4.py
python TEM-MDM.py
```

These scripts generate both point-level inputs and patch-level inputs.

## Pretrained Checkpoint

This repository includes a small pretrained HybridCNNLSTM checkpoint at:

```text
code/model_save/hybrid_cnn_lstm/base_model.pth
```

This checkpoint was obtained from TEM-MDM temporal pretraining and corresponds to the best validation checkpoint used for FLUXNET finetuning. It is provided for convenience so that users can directly run pretrained FLUXNET finetuning with `--load_pretrain` without rerunning TEM-MDM pretraining first.

To use the checkpoint, run:

```bash
cd code/model_training

python finetune.py \
  --valid_type temporal \
  --model hybrid_cnn_lstm \
  --id pretrained_realpatch \
  --epoch 30 \
  --lr 0.001 \
  --load_pretrain
```

The checkpoint is small and included only for reproducibility. Other generated model checkpoints, logs, raw data files, and processed NumPy arrays are not included in this repository.

## Training

### 1. TEM-MDM Pretraining

```bash
cd code/model_training

python pretrain.py \
  --valid_type temporal \
  --model hybrid_cnn_lstm \
  --epoch 30 \
  --id run2_best \
  --lr 0.003
```

This saves the best pretrained checkpoint as:

```text
../model_save/hybrid_cnn_lstm/base_model.pth
```

The provided `base_model.pth` checkpoint corresponds to this TEM-MDM pretraining result.

### 2. FLUXNET Temporal Finetuning

```bash
python finetune.py \
  --valid_type temporal \
  --model hybrid_cnn_lstm \
  --id pretrained_realpatch \
  --epoch 30 \
  --lr 0.001 \
  --load_pretrain
```

### 3. FLUXNET Spatial Finetuning

```bash
python finetune.py \
  --valid_type spatial \
  --model hybrid_cnn_lstm \
  --id pretrained_realpatch \
  --epoch 30 \
  --lr 0.001 \
  --load_pretrain
```

## Experimental Results

Main results from our experiments:

| Experiment | RMSE | R² |
|---|---:|---:|
| Scratch, FLUXNET temporal | 31.39 | -0.434 |
| Scratch, FLUXNET spatial | 82.97 | -0.284 |
| Pretrain, TEM-MDM temporal | 63.74 | 0.968 |
| Pretrained, FLUXNET temporal | 18.24 | 0.516 |
| Pretrained, FLUXNET spatial | 75.73 | -0.070 |

TEM-MDM pretraining substantially improves FLUXNET temporal finetuning and also improves spatial finetuning, although spatial generalization to unseen FLUXNET sites remains challenging.

## Notes

This repository is an extension of the original X-MethaneWet codebase. The main difference is the addition of point-plus-patch dual-input learning for HybridCNNLSTM.

The original workflow uses a single-input format:

```text
(x, y)
```

This project adds a dual-input format for HybridCNNLSTM:

```text
(x_patch, x_point, y)
```

The included `base_model.pth` is the only checkpoint intentionally tracked in this repository. Other checkpoints should be regenerated through the training commands above.

## Citation

If you use the original X-MethaneWet dataset, please cite:

```bibtex
@article{sun2025x,
  title={X-MethaneWet: A Cross-scale Global Wetland Methane Emission Benchmark Dataset for Advancing Science Discovery with AI},
  author={Sun, Yiming and Chen, Shuo and Chen, Shengyu and Qiu, Chonghao and Liu, Licheng and Oh, Youmi and Malone, Sparkle L and McNicol, Gavin and Zhuang, Qianlai and Smith, Chris and Xie, Yiqun and Jia, Xiaowei},
  journal={arXiv preprint arXiv:2505.18355},
  year={2025}
}
```

## Acknowledgement

This project builds on the original X-MethaneWet dataset and codebase. Our contribution is the patch-augmented preprocessing and HybridCNNLSTM training pipeline for point-plus-patch methane emission upscaling.

## Contact

For questions about this project extension, please contact:

Xiaoyan Wei  
xiw249@pitt.edu
