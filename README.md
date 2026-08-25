# Brain Tumor Classification

Reproducible training, evaluation, and inference tooling for the Kaggle brain tumor image classification dataset: [prathamgrover/brain-tumor-classification](https://www.kaggle.com/datasets/prathamgrover/brain-tumor-classification).

This repository modernizes the original ResNet50 experiment into a package with:

- explicit CPU and CUDA install paths
- a deterministic train/validation split created from `data/Training`
- a held-out final evaluation on `data/Testing`
- duplicate-hash reporting across training and test data
- metadata-rich checkpoints with SHA-256 hashes
- offline unit and smoke tests
- a documented `brain-tumor` CLI for download, inspection, training, evaluation, and prediction

## Important non-clinical disclaimer

This project is an educational computer-vision benchmark. It is **not** a medical device and must not be used for diagnosis, triage, treatment planning, prognosis, or clinical decision-making.

## Classes

The classifier predicts four labels:

- `glioma`
- `meningioma`
- `notumor`
- `pituitary`

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Aeryes/BrainTumorClassification.git
cd BrainTumorClassification
```

### 2. Install PyTorch

CPU:

```bash
python -m pip install -r requirements/torch-cpu.txt
```

CUDA 11.8:

```bash
python -m pip install -r requirements/torch-cu118.txt
```

### 3. Install the project

```bash
python -m pip install -e ".[dev]"
```

The project intentionally keeps `torch` and `torchvision` out of `pyproject.toml` because the correct wheel depends on your local CPU/GPU and Python version. Install them first from the matching requirements file, then install the package itself.

### 4. Download or validate the dataset

If Kaggle credentials are already configured:

```bash
brain-tumor data download --data-root data --seed 42
```

If the dataset is already extracted locally:

```bash
brain-tumor data inspect --data-root data --seed 42
```

This generates:

- `data/manifests/train_validation_split.csv`
- `configs/brain-tumor.dataset.json`

The modernized training flow creates a deterministic train/validation split from `data/Training` and reserves `data/Testing` for held-out evaluation only.

## Train

The default baseline preserves the original transfer-learning head while fixing scheduler and checkpoint behavior:

```bash
brain-tumor train --config configs/train.toml
```

Useful overrides:

```bash
brain-tumor train --config configs/train.toml --device cuda --seed 42
brain-tumor train --config configs/train.toml --device cpu --dry-run --no-pretrained-weights
```

Artifacts are written to `artifacts/runs/<run-id>/`, including:

- best and last checkpoints
- training curves
- training history
- checkpoint metadata with SHA-256
- resolved run metadata

## Evaluate

Evaluate the best checkpoint in a single final pass against Kaggle `Testing`, excluding leaked duplicates from the reported result:

```bash
brain-tumor evaluate \
  --checkpoint artifacts/runs/<run-id>/checkpoints/best.pt \
  --test-root data/Testing \
  --output-dir artifacts/runs/<run-id>/metrics \
  --exclude-leaked-test-images \
  --train-root data/Training
```

This writes:

- `test_metrics.json`
- `confusion_matrix.png`
- `checkpoint_metadata.json`

## Predict

Predict a single image:

```bash
brain-tumor predict --checkpoint artifacts/runs/<run-id>/checkpoints/best.pt --image path/to/image.jpg
```

Predict every image in the default inference folder and export JSON:

```bash
brain-tumor predict \
  --checkpoint artifacts/runs/<run-id>/checkpoints/best.pt \
  --folder data/New \
  --json-output artifacts/runs/<run-id>/predictions.json
```

Legacy compatibility scripts remain available:

```bash
python train.py
python predict.py
python visualize_and_analyze.py
```

## Repository layout

```text
src/brain_tumor_classification/
  cli.py
  data.py
  training.py
  evaluation.py
  inference.py
  checkpoints.py
configs/
  train.toml
  brain-tumor.dataset.json
tests/
.github/workflows/ci.yml
```

## Historical experiment artifacts

The original repository included visual artifacts from the earlier experiment:

### Training data without augmentation
![Training Data No Transforms](/images/train_data_no_transforms.png)

### Training data with augmentation
![Training Data With Transforms](/images/train_data_with_transforms.png)

### Validation data without augmentation
![Validation Data No Transforms](/images/validation_data_no_transforms.png)

### Historical training curves
![Training Results](/images/Figure_1.png)

### Historical confusion matrix
![Confusion Matrix](/images/confusion_matrix.png)

### Historical prediction examples
![Predicted Image One](/images/predicted_one.png)
![Predicted Image Two](/images/predicted_two.png)

## Results and comparison

The original README reported `98.77%` accuracy. Treat that number as a historical experiment result, not a verified benchmark for the modernized workflow, because the earlier process reused Kaggle `Testing` during development. Verified metrics from new runs should come only from:

```bash
brain-tumor train ...
brain-tumor evaluate ...
```

and be recorded under `artifacts/runs/<run-id>/metrics/`.

Verified baseline run (`artifacts/runs/20260825-214008/`):

- device used: CUDA (`NVIDIA GeForce RTX 4080 SUPER`)
- training source: deterministic split from `data/Training`
- final evaluation protocol: Kaggle `Testing` with `88` leaked images excluded, leaving a clean subset of `306` images
- clean held-out accuracy: `52.94%`
- macro precision / recall / F1: `64.81%` / `62.11%` / `49.49%`
- weighted precision / recall / F1: `75.62%` / `52.94%` / `54.85%`
- best checkpoint SHA-256: `57edc066e53cf84cbdffdfe23fc4496af6742ce201b76716677ad0f6d99e34ab`

This verified run is dramatically lower than the historical `98.77%` claim. That difference is expected because the historical workflow repeatedly observed Kaggle `Testing`, while the modernized evaluation reports only a single final pass on the leak-filtered held-out subset.

## Limitations

- Duplicate hashes across `Training` and `Testing` must be reviewed before trusting held-out metrics.
- Patient-level independence cannot be inferred solely from the directory structure.
- Historical and modernized metrics are not directly comparable if the split or preprocessing changes.
- Confidence scores are not calibrated probabilities for clinical or operational use.
- The committed legacy `resnet_model.pth` should be treated as a historical artifact until a release-hosted checkpoint with metadata is published.

## Development checks

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m pytest --cov=brain_tumor_classification
python -m build
python -m twine check dist/*
```

## Documentation

- [MODEL_CARD.md](MODEL_CARD.md)
- [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
- [LICENSE](LICENSE)
