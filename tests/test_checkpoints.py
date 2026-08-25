from __future__ import annotations

from pathlib import Path

import torch

from brain_tumor_classification.checkpoints import load_model_from_checkpoint, save_checkpoint
from brain_tumor_classification.constants import CLASS_NAMES
from brain_tumor_classification.modeling import ModelConfig, build_model


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model_config = ModelConfig(load_pretrained_weights=False)
    model = build_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    checkpoint_path = tmp_path / "checkpoint.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        metrics={"validation_loss": 0.5, "validation_accuracy": 50.0},
        class_names=CLASS_NAMES,
        model_config=ModelConfig(load_pretrained_weights=False),
        training_config={"seed": 42},
        history={
            "train_loss": [1.0],
            "validation_loss": [0.5],
            "train_accuracy": [50.0],
            "validation_accuracy": [50.0],
        },
    )

    loaded_model, payload = load_model_from_checkpoint(checkpoint_path)
    assert payload["format_version"] == 1
    assert payload["class_names"] == list(CLASS_NAMES)
    assert isinstance(loaded_model, torch.nn.Module)
