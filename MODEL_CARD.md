# Model Card: Brain Tumor Classification

## Summary
This repository trains a ResNet50 transfer-learning classifier for four MRI image classes from the Kaggle brain tumor dataset:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Intended use
- Educational computer-vision experiments
- Reproducible training and evaluation workflows
- Portfolio demonstration of dataset handling, transfer learning, and offline inference

## Explicit non-clinical scope
This repository is **not** a medical device and must not be used for:

- diagnosis
- triage
- treatment planning
- prognosis
- clinical decision-making

## Architecture
- Backbone: torchvision `resnet50`
- Transfer-learning head:
  - `Linear(in_features, 256)`
  - `ReLU`
  - `Dropout(0.5)`
  - `Linear(256, 4)`

## Data
- Source: Kaggle dataset `prathamgrover/brain-tumor-classification`
- Held-out test split: `data/Testing`
- Deterministic train/validation split source: `data/Training`
- Local dataset metadata and duplicate reports are stored in `configs/brain-tumor.dataset.json` after `brain-tumor data download` or `brain-tumor data inspect`

## Preprocessing
- Train: resize, horizontal flip, rotation, color jitter, affine shear, ImageNet normalization
- Eval: resize and ImageNet normalization

## Metrics
- Historical repository README claim: `98.77%` accuracy
- That historical number was measured under a workflow that repeatedly observed the Kaggle `Testing` split during development and should not be treated as a blinded clinical estimate
- Verified metrics from modernized runs are written to `artifacts/runs/<run-id>/metrics/test_metrics.json`
- Verified baseline run: `artifacts/runs/20260825-214008/metrics/test_metrics.json`
  - evaluation protocol: exclude `88` leaked Kaggle `Testing` images and report metrics on the remaining `306` clean images
  - clean held-out accuracy: `52.94%`
  - macro precision / recall / F1: `64.81%` / `62.11%` / `49.49%`
  - weighted precision / recall / F1: `75.62%` / `52.94%` / `54.85%`
  - best checkpoint SHA-256: `57edc066e53cf84cbdffdfe23fc4496af6742ce201b76716677ad0f6d99e34ab`

## Reproducibility
- Default seed: `42`
- Best and last checkpoints include class names, config, epoch, metrics, and training history
- Checkpoint metadata includes SHA-256 hashes

## Limitations
- Patient-level independence cannot be guaranteed from directory structure alone
- Duplicate hashes across `Training` and `Testing` must be reviewed before trusting held-out metrics
- Performance depends on the exact Kaggle download contents and deterministic split seed
- This benchmark is not a substitute for clinical validation across scanners, institutions, protocols, or populations
