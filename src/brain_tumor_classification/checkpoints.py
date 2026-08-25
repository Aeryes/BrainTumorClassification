from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from brain_tumor_classification.constants import CLASS_NAMES
from brain_tumor_classification.modeling import ModelConfig, build_model


def sha256_path(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_torch_file(path: Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def save_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau,
    epoch: int,
    metrics: dict[str, float],
    class_names: tuple[str, ...] = CLASS_NAMES,
    model_config: ModelConfig,
    training_config: dict[str, object],
    history: dict[str, list[float]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "class_names": list(class_names),
        "model_config": model_config.to_dict(),
        "training_config": training_config,
        "history": history,
    }
    torch.save(payload, path)
    return path


def save_checkpoint_metadata(checkpoint_path: Path, output_path: Path) -> Path:
    metadata = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "size_bytes": checkpoint_path.stat().st_size,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_path


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_torch_file(checkpoint_path, map_location=map_location)

    if isinstance(payload, dict) and "model_state_dict" in payload:
        model_config = ModelConfig(**payload.get("model_config", {}))
        model = build_model(
            ModelConfig(
                model_name=model_config.model_name,
                num_classes=model_config.num_classes,
                dropout_rate_1=model_config.dropout_rate_1,
                dropout_rate_2=model_config.dropout_rate_2,
                load_pretrained_weights=False,
            )
        )
        model.load_state_dict(payload["model_state_dict"])
        return model, payload

    model = build_model(ModelConfig(load_pretrained_weights=False))
    model.load_state_dict(payload)
    return model, {
        "format_version": 0,
        "class_names": list(CLASS_NAMES),
        "model_config": ModelConfig(load_pretrained_weights=False).to_dict(),
    }


def save_legacy_state_dict(path: Path, model: torch.nn.Module) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
