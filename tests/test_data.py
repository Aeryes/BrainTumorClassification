from __future__ import annotations

from pathlib import Path

from brain_tumor_classification.constants import CLASS_NAMES
from brain_tumor_classification.data import (
    build_duplicate_report,
    create_split_manifest,
    inspect_dataset,
    read_split_manifest,
)

IMAGES_PER_CLASS = 3


def test_inspect_dataset_counts_classes(tiny_brain_dataset: Path) -> None:
    summary = inspect_dataset(tiny_brain_dataset / "Training")
    assert summary["total_images"] == len(CLASS_NAMES) * IMAGES_PER_CLASS
    assert summary["class_counts"]["glioma"] == IMAGES_PER_CLASS


def test_split_manifest_is_deterministic(tiny_brain_dataset: Path, tmp_path: Path) -> None:
    manifest_one = create_split_manifest(
        train_root=tiny_brain_dataset / "Training",
        output_path=tmp_path / "one.csv",
        seed=42,
    )
    manifest_two = create_split_manifest(
        train_root=tiny_brain_dataset / "Training",
        output_path=tmp_path / "two.csv",
        seed=42,
    )
    assert manifest_one.read_text(encoding="utf-8") == manifest_two.read_text(encoding="utf-8")


def test_read_split_manifest_returns_train_and_validation(
    tiny_brain_dataset: Path,
    tmp_path: Path,
) -> None:
    manifest = create_split_manifest(
        train_root=tiny_brain_dataset / "Training",
        output_path=tmp_path / "split.csv",
        seed=42,
    )
    splits = read_split_manifest(manifest, tiny_brain_dataset / "Training")
    assert splits["train"]
    assert splits["validation"]


def test_duplicate_report_is_clean_for_synthetic_data(tiny_brain_dataset: Path) -> None:
    report = build_duplicate_report(
        train_root=tiny_brain_dataset / "Training",
        test_root=tiny_brain_dataset / "Testing",
    )
    assert report["duplicate_group_count"] == 0
