from __future__ import annotations

from pathlib import Path

from brain_tumor_classification.evaluation import evaluate_checkpoint
from brain_tumor_classification.inference import predict_folder
from brain_tumor_classification.training import TrainConfig, train_model


def test_train_evaluate_predict_smoke(tiny_brain_dataset: Path, tmp_path: Path) -> None:
    result = train_model(
        TrainConfig(
            data_root=tiny_brain_dataset,
            output_root=tmp_path / "runs",
            split_manifest=tmp_path / "split.csv",
            device="cpu",
            epochs=1,
            batch_size=2,
            dry_run=True,
            load_pretrained_weights=False,
        )
    )
    checkpoint_path = Path(result["best_checkpoint"])
    assert checkpoint_path.exists()

    metrics = evaluate_checkpoint(
        checkpoint_path,
        test_root=tiny_brain_dataset / "Testing",
        output_dir=tmp_path / "evaluation",
        batch_size=2,
        device_name="cpu",
    )
    assert "accuracy" in metrics
    assert metrics["per_class"]["glioma"]["sensitivity"] == metrics["per_class"]["glioma"]["recall"]
    assert metrics["per_class"]["glioma"]["specificity"] is not None

    predictions = predict_folder(
        checkpoint_path,
        folder_path=tiny_brain_dataset / "New",
        device_name="cpu",
        output_path=tmp_path / "predictions.json",
    )
    assert predictions
