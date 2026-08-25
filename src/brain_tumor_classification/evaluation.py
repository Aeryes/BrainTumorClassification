from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from brain_tumor_classification.checkpoints import (
    load_model_from_checkpoint,
    save_checkpoint_metadata,
)
from brain_tumor_classification.constants import CLASS_NAMES, DEFAULT_TEST_DIR
from brain_tumor_classification.data import (
    BrainTumorDataset,
    build_clean_test_items,
    enumerate_labeled_items,
)
from brain_tumor_classification.training import resolve_device
from brain_tumor_classification.transforms import build_eval_transform


def _build_per_class_metrics(
    report: dict[str, object],
    matrix: np.ndarray,
    class_names: list[str],
) -> dict[str, dict[str, float | int | None]]:
    total = int(matrix.sum())
    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, class_name in enumerate(class_names):
        class_report = cast(dict[str, float | int | None], report[class_name]).copy()
        true_positive = int(matrix[index, index])
        false_negative = int(matrix[index, :].sum() - true_positive)
        false_positive = int(matrix[:, index].sum() - true_positive)
        true_negative = total - true_positive - false_negative - false_positive
        specificity_denominator = true_negative + false_positive
        specificity = (
            true_negative / specificity_denominator if specificity_denominator > 0 else None
        )
        class_report["sensitivity"] = class_report["recall"]
        class_report["specificity"] = specificity
        per_class[class_name] = class_report
    return per_class


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    test_root: Path = DEFAULT_TEST_DIR,
    output_dir: Path | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    device_name: str = "auto",
    exclude_leaked_test_images: bool = False,
    train_root: Path | None = None,
) -> dict[str, object]:
    device = resolve_device(device_name)
    model, metadata = load_model_from_checkpoint(checkpoint_path, map_location=device)
    model.to(device)
    model.eval()

    leakage_report: dict[str, object] | None = None
    if exclude_leaked_test_images:
        effective_train_root = train_root or Path("data") / "Training"
        test_items, leakage_report = build_clean_test_items(
            train_root=effective_train_root,
            test_root=test_root,
        )
    else:
        test_items = enumerate_labeled_items(test_root)

    dataset = BrainTumorDataset(test_items, transform=build_eval_transform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            features = batch_features.to(device)
            labels = batch_labels.to(device)
            outputs = model(features)
            predictions = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(predictions.cpu().tolist())

    class_names = metadata.get("class_names", list(CLASS_NAMES))
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred)
    per_class = _build_per_class_metrics(report, matrix, class_names)

    result = {
        "checkpoint_path": str(checkpoint_path),
        "class_names": class_names,
        "accuracy": report["accuracy"],
        "macro_avg": report["macro avg"],
        "weighted_avg": report["weighted avg"],
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "excluded_leakage": leakage_report,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "test_metrics.json"
        metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        fig, axis = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axis,
        )
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title("Confusion Matrix")
        fig.tight_layout()
        fig.savefig(output_dir / "confusion_matrix.png")
        plt.close(fig)
        save_checkpoint_metadata(checkpoint_path, output_dir / "checkpoint_metadata.json")

    return result
