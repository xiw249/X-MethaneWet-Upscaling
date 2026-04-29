# X-MethaneWet-Upscaling
A patch-augmented HybridCNNLSTM pipeline for wetland methane emission upscaling using X-MethaneWet, TEM-MDM simulation data, and FLUXNET-CH4 observations.


## Overview

**X-MethaneWet** is the first cross-scale global wetland methane benchmark dataset, synthesizing physics-based model simulation data (TEM-MDM) and real-world observation data (FLUXNET-CH₄). It offers a comprehensive foundation for applying machine learning techniques to CH₄ flux modeling, enabling robust evaluation under diverse spatial and temporal conditions.

## Features

* Combines simulated and observed methane flux data for robust benchmarking
* Supports a variety of deep learning models (LSTM, EA-LSTM, TCN, Transformer, iTransformer, Pyraformer, etc.)
* Includes multiple transfer learning techniques (adversarial, residual, reweighting, fine-tuning)
* Modular codebase for data processing, model training, and evaluation

## Repository Structure
```
data/                          # Download dataset from https://huggingface.co/datasets/ymsun99/X-MethaneWet and put it under the folder
code/
├── data_processing/
│   ├── FLUXNET-CH4.py      # Preprocessing for FLUXNET methane dataset
│   └── TEM-MDM.py          # Preprocessing for TEM-MDM dataset
│
├── model_training/
│   ├── adversarial.py         # Adversarial transfer learning implementation
│   ├── base_model.py          # Base model for parameter transfer
│   ├── config.py              # Configuration settings
│   ├── finetune.py            # Fine-tuning and training from scratch
│   ├── model.py               # Core model architectures
│   ├── pretrain.py            # Pretraining on TEM-MDM dataset
│   ├── residual.py            # Residual learning method
│   └── reweight.py            # Reweighting method
│
└── README.md
```

## Setup & Requirements

Before starting, please clone the required Time Series Library into the `model_training/` directory:

````bash
git clone https://github.com/thuml/Time-Series-Library.git
````

1. **Process dataset:**

   ```bash
   cd code/data_processing
   python FLUXNET-CH4.py
   python TEM-MDM.py
   ```
2. **Train models:**

   ```bash
   cd code/model_training

   # Train on TEM-MDM data
   python pretrain.py --valid_type temporal --model hybrid_cnn_lstm --epoch 30 --id run2_best --lr 0.003

   # Fine-tune on FLUXNET-CH4 data
   python finetune.py --valid_type temporal --model hybrid_cnn_lstm --id pretrained_realpatch --epoch 30 --lr 0.001 --load_pretrain
   python finetune.py --valid_type spatial --model hybrid_cnn_lstm --id pretrained_realpatch --epoch 30 --lr 0.001 --load_pretrain
   

   # Residual modeling 
   python residual.py --valid_type temporal --model lstm --id run1 --epoch 200 --lr 0.02
   python residual.py --valid_type spatial --model lstm --epoch 200 --spatial_fold 0 --lr 0.02

   # Adversarial learning
   python DANN.py --valid_type temporal --model lstm --id run1 --epoch 200 --lr 0.02
   python DANN.py --valid_type spatial --model lstm --epoch 200 --spatial_fold 0 --lr 0.02

   # Reweight data
   python reweight.py --valid_type temporal --model lstm --id run1 --epoch 200 --lr 0.02
   python reweight.py --valid_type spatial --model lstm --epoch 200 --spatial_fold 0 --lr 0.02
   ```

Replace `lstm` with your desired model (e.g., `transformer`, `iTransformer`, etc.).

## Citation

If you use X-MethaneWet or this codebase in your research, please cite:

```bibtex
@article{sun2025x,
  title={X-MethaneWet: A Cross-scale Global Wetland Methane Emission Benchmark Dataset for Advancing Science Discovery with AI},
  author={Sun, Yiming and Chen, Shuo and Chen, Shengyu and Qiu, Chonghao and Liu, Licheng and Oh, Youmi and Malone, Sparkle L and McNicol, Gavin and Zhuang, Qianlai and Smith, Chris and Xie, Yiqun and Jia, Xiaowei},
  journal={arXiv preprint arXiv:2505.18355},
  year={2025}
}
```

## Contact

For questions or contributions, please feel free to contact Yiming Sun at yimingsun@pitt.edu.
