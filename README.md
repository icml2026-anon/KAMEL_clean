# KAMEL: Knowledge-Aware Multi-modal Ensemble Learning

This repository contains the code and datasets for the paper submitted to ICML 2026.

## Overview

KAMEL is a tabular classification framework combining KAN (Kolmogorov-Arnold Networks) encoders with a hierarchical Mixture-of-Experts (MoE) fusion module.

## Repository Structure

```
KAMEL_clean/
├── kamel/                   # Main model package
│   ├── models/              # KAN encoders, MoE fusion, KAMEL model
│   ├── data/                # Data preprocessing and transforms
│   ├── training/            # Trainer, losses, imbalance handling
│   └── utils/               # Config, metrics, visualization
├── model_comparison/        # Baseline model implementations
│   ├── algorithms/          # CatBoost, XGBoost, LightGBM, TabNet, FT-Transformer, SAINT, TabR, NODE, GANDALF, TabPFN, TabICL, MotherNet, TabM, ...
│   ├── ml_experiment.py     # Baseline experiment runner
│   └── algorithm_configs.py # Hyperparameter configs for all baselines
├── data/                    # 25 benchmark datasets (CSV)
├── train_kamel.py           # KAMEL training entry point
└── run_ml_comparison.py     # Baseline comparison entry point
```

## Datasets

25 benchmark datasets covering binary and multiclass classification tasks from the UCI repository.

| Task | Datasets |
|------|----------|
| Binary (13) | Banknote, Blood Transfusion, Breast Cancer, Diabetes, Haberman, Heart Disease, Hepatitis, Liver Disorders, Mammographic, MONK's-1, Parkinsons, Spambase, Vertebral Column |
| Multiclass (12) | Authorship, Car Evaluation, Cardiotocography, CMC, MFeat-Fourier, MFeat-Morphological, MFeat-Zernike, Nursery, Solar Flare, Vehicle, Wine Quality, Yeast |

## Usage

### Train KAMEL

```bash
python train_kamel.py \
  --dataset banknote \
  --data_path data/banknote.csv \
  --model_size base \
  --epochs 200 \
  --device cuda
```

### Run Baseline Comparison

```bash
python run_ml_comparison.py \
  --dataset banknote \
  --algorithm all
```

## Requirements

See `model_comparison/` for baseline dependencies. Core KAMEL dependencies:

```
torch
numpy
scikit-learn
pandas
imbalanced-learn
optuna
```
