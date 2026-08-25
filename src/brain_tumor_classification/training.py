from __future__ import annotations

import json
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from brain_tumor_classification.checkpoints import (
    save_checkpoint,
    save_checkpoint_metadata,
)
from brain_tumor_classification.constants import (
    CLASS_NAMES,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
)
from brain_tumor_classification.data import (
    BrainTumorDataset,
    ensure_split_manifest,
    read_split_manifest,
)
from brain_tumor_classification.modeling import ModelConfig, build_model
from brain_tumor_classification.transforms import (
    build_eval_transform,
    build_train_transform,
)


@dataclass(frozen=True)
class TrainConfig:
    data_root: Path = Path("data")
    output_root: Path = Path("artifacts") / "runs"
    split_manifest: Path = Path("data") / "manifests" / "train_validation_split.csv"
    seed: int = 42
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    num_workers: int = 0
    learning_rate_backbone: float = 1e-5
    learning_rate_head: float = 1e-4
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    early_stopping_patience: int = 3
    device: str = "auto"
    use_amp: bool = True
    image_size: int = 224
    dry_run: bool = False
    load_pretrained_weights: bool = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _compute_class_weights(dataset: BrainTumorDataset, device: torch.device) -> torch.Tensor:
    labels = [item.label for item in dataset.items]
    classes = np.array(list(range(len(CLASS_NAMES))))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _build_run_dir(output_root: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _plot_history(history: dict[str, list[float]], output_path: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history["train_loss"], label="Train Loss")
    ax1.plot(history["validation_loss"], label="Validation Loss")
    ax1.set_title("Loss over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(history["train_accuracy"], label="Train Accuracy")
    ax2.plot(history["validation_accuracy"], label="Validation Accuracy")
    ax2.set_title("Accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _write_run_metadata(run_dir: Path, config: TrainConfig, device: torch.device) -> Path:
    metadata = {
        "config": asdict(config),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pid": os.getpid(),
    }
    output_path = run_dir / "run_metadata.json"
    output_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return output_path


def train_model(config: TrainConfig) -> dict[str, object]:
    seed_everything(config.seed)
    device = resolve_device(config.device)

    manifest_path = ensure_split_manifest(
        train_root=config.data_root / "Training",
        manifest_path=config.split_manifest,
        seed=config.seed,
        validation_fraction=config.validation_fraction,
    )
    split_items = read_split_manifest(manifest_path, config.data_root / "Training")

    train_dataset = BrainTumorDataset(
        split_items["train"],
        transform=build_train_transform(config.image_size),
    )
    validation_dataset = BrainTumorDataset(
        split_items["validation"],
        transform=build_eval_transform(config.image_size),
    )

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    model_config = ModelConfig(load_pretrained_weights=config.load_pretrained_weights)
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(
        [
            {"params": model.resnet.layer4.parameters(), "lr": config.learning_rate_backbone},
            {"params": model.resnet.fc.parameters(), "lr": config.learning_rate_head},
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=2,
    )
    loss_fn = torch.nn.CrossEntropyLoss(weight=_compute_class_weights(train_dataset, device))

    run_dir = _build_run_dir(config.output_root)
    checkpoints_dir = run_dir / "checkpoints"
    metrics_dir = run_dir / "metrics"
    _write_run_metadata(run_dir, config, device)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_validation_loss = float("inf")
    best_checkpoint_path = checkpoints_dir / "best.pt"
    last_checkpoint_path = checkpoints_dir / "last.pt"
    early_stop_counter = 0

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and config.use_amp,
    )

    epochs_to_run = 1 if config.dry_run else config.epochs
    for epoch in range(epochs_to_run):
        model.train()
        running_train_loss = 0.0
        num_correct_train = 0
        num_predictions_train = 0

        for batch_features, batch_labels in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs_to_run} - Training",
        ):
            train_features = batch_features.to(device)
            train_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda",
                enabled=device.type == "cuda" and config.use_amp,
            ):
                predictions = model(train_features)
                loss = loss_fn(predictions, train_labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_train_loss += loss.item()
            predicted = predictions.argmax(dim=1)
            num_correct_train += (predicted == train_labels).sum().item()
            num_predictions_train += train_labels.size(0)

        average_train_loss = running_train_loss / max(len(train_loader), 1)
        average_train_accuracy = (num_correct_train / max(num_predictions_train, 1)) * 100
        history["train_loss"].append(average_train_loss)
        history["train_accuracy"].append(average_train_accuracy)

        model.eval()
        running_validation_loss = 0.0
        num_correct_validation = 0
        num_predictions_validation = 0
        with torch.no_grad():
            for batch_validation_features, batch_validation_labels in tqdm(
                validation_loader,
                desc=f"Epoch {epoch + 1}/{epochs_to_run} - Validating",
            ):
                validation_features = batch_validation_features.to(device)
                validation_labels = batch_validation_labels.to(device)
                predictions = model(validation_features)
                loss = loss_fn(predictions, validation_labels)

                running_validation_loss += loss.item()
                predicted = predictions.argmax(dim=1)
                num_correct_validation += (predicted == validation_labels).sum().item()
                num_predictions_validation += validation_labels.size(0)

        average_validation_loss = running_validation_loss / max(len(validation_loader), 1)
        average_validation_accuracy = (
            num_correct_validation / max(num_predictions_validation, 1)
        ) * 100
        history["validation_loss"].append(average_validation_loss)
        history["validation_accuracy"].append(average_validation_accuracy)
        scheduler.step(average_validation_loss)

        epoch_metrics = {
            "train_loss": average_train_loss,
            "train_accuracy": average_train_accuracy,
            "validation_loss": average_validation_loss,
            "validation_accuracy": average_validation_accuracy,
        }
        save_checkpoint(
            path=last_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,
            metrics=epoch_metrics,
            class_names=CLASS_NAMES,
            model_config=ModelConfig(load_pretrained_weights=False),
            training_config=asdict(config),
            history=history,
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            early_stop_counter = 0
            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch + 1,
                metrics=epoch_metrics,
                class_names=CLASS_NAMES,
                model_config=ModelConfig(load_pretrained_weights=False),
                training_config=asdict(config),
                history=history,
            )
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stopping_patience and not config.dry_run:
                break

    history_path = metrics_dir / "training_history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    curves_path = _plot_history(history, metrics_dir / "training_curves.png")
    save_checkpoint_metadata(best_checkpoint_path, metrics_dir / "best_checkpoint.json")

    split_counts = {
        name: dict(Counter(item.class_name for item in items))
        for name, items in split_items.items()
    }

    return {
        "run_dir": run_dir,
        "best_checkpoint": best_checkpoint_path,
        "last_checkpoint": last_checkpoint_path,
        "history_path": history_path,
        "curves_path": curves_path,
        "manifest_path": manifest_path,
        "device": str(device),
        "split_counts": split_counts,
    }
