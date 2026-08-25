from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image
from torch.utils.data import Dataset

from brain_tumor_classification.constants import (
    CLASS_NAMES,
    DATASET_SLUG,
    DEFAULT_DATA_ROOT,
    DEFAULT_TEST_DIR,
    DEFAULT_TRAIN_DIR,
    DIRECTORY_NAMES,
    IMAGE_EXTENSIONS,
)


@dataclass(frozen=True)
class DatasetItem:
    path: Path
    label: int
    class_name: str


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _iter_class_files(root: Path, class_name: str) -> list[Path]:
    class_dir = root / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(f"Missing class directory: {class_dir}")

    files = sorted(
        file_path
        for file_path in class_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No supported images found in {class_dir}")
    return files


def enumerate_labeled_items(root: Path) -> list[DatasetItem]:
    items: list[DatasetItem] = []
    for label, (class_name, directory_name) in enumerate(
        zip(CLASS_NAMES, DIRECTORY_NAMES, strict=True)
    ):
        for file_path in _iter_class_files(root, directory_name):
            items.append(DatasetItem(path=file_path, label=label, class_name=class_name))
    return items


class BrainTumorDataset(Dataset):
    def __init__(self, items: Iterable[DatasetItem], transform=None) -> None:  # type: ignore[no-untyped-def]
        self.items = list(items)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):  # type: ignore[no-untyped-def]
        item = self.items[idx]
        image = Image.open(item.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item.label


def load_data(dataset_path: str | Path, transform=None) -> BrainTumorDataset:  # type: ignore[no-untyped-def]
    return BrainTumorDataset(enumerate_labeled_items(Path(dataset_path)), transform=transform)


def inspect_dataset(root: Path) -> dict[str, object]:
    items = enumerate_labeled_items(root)
    counter = Counter(item.class_name for item in items)
    return {
        "root": str(root),
        "total_images": len(items),
        "class_counts": dict(counter),
    }


def build_duplicate_report(
    train_root: Path = DEFAULT_TRAIN_DIR, test_root: Path = DEFAULT_TEST_DIR
) -> dict[str, object]:
    seen_hashes: dict[str, list[str]] = {}
    for root_name, root_path in (("Training", train_root), ("Testing", test_root)):
        for item in enumerate_labeled_items(root_path):
            file_hash = sha256_file(item.path)
            seen_hashes.setdefault(file_hash, []).append(f"{root_name}/{item.path.name}")

    duplicates = {file_hash: paths for file_hash, paths in seen_hashes.items() if len(paths) > 1}
    cross_split_groups = {
        file_hash: paths
        for file_hash, paths in duplicates.items()
        if any(path.startswith("Training/") for path in paths)
        and any(path.startswith("Testing/") for path in paths)
    }
    return {
        "duplicate_group_count": len(duplicates),
        "cross_split_group_count": len(cross_split_groups),
        "duplicate_groups": duplicates,
        "cross_split_groups": cross_split_groups,
    }


def build_clean_test_items(
    *,
    train_root: Path = DEFAULT_TRAIN_DIR,
    test_root: Path = DEFAULT_TEST_DIR,
) -> tuple[list[DatasetItem], dict[str, object]]:
    train_hashes = {sha256_file(item.path) for item in enumerate_labeled_items(train_root)}
    clean_items: list[DatasetItem] = []
    excluded_paths: list[str] = []

    for item in enumerate_labeled_items(test_root):
        item_hash = sha256_file(item.path)
        if item_hash in train_hashes:
            excluded_paths.append(str(item.path))
            continue
        clean_items.append(item)

    report = {
        "excluded_test_images": len(excluded_paths),
        "remaining_test_images": len(clean_items),
        "excluded_paths": excluded_paths,
    }
    return clean_items, report


def create_split_manifest(
    train_root: Path = DEFAULT_TRAIN_DIR,
    output_path: Path | None = None,
    *,
    seed: int,
    validation_fraction: float = 0.2,
) -> Path:
    if output_path is None:
        output_path = DEFAULT_DATA_ROOT / "manifests" / "train_validation_split.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    rows: list[dict[str, object]] = []

    for label, (class_name, directory_name) in enumerate(
        zip(CLASS_NAMES, DIRECTORY_NAMES, strict=True)
    ):
        files = _iter_class_files(train_root, directory_name)
        shuffled = files[:]
        rng.shuffle(shuffled)
        validation_count = max(1, int(round(len(shuffled) * validation_fraction)))

        for index, file_path in enumerate(shuffled):
            split = "validation" if index < validation_count else "train"
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "class_name": class_name,
                    "relative_path": str(file_path.relative_to(train_root)).replace("\\", "/"),
                    "sha256": sha256_file(file_path),
                }
            )

    rows.sort(
        key=lambda row: (
            str(row["split"]),
            str(row["class_name"]),
            str(row["relative_path"]),
        )
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "label", "class_name", "relative_path", "sha256"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def read_split_manifest(
    manifest_path: Path,
    train_root: Path = DEFAULT_TRAIN_DIR,
) -> dict[str, list[DatasetItem]]:
    splits: dict[str, list[DatasetItem]] = {"train": [], "validation": []}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            split = row["split"]
            if split not in splits:
                raise ValueError(f"Unexpected split name {split!r} in {manifest_path}")
            relative_path = Path(row["relative_path"])
            class_name = row["class_name"]
            label = int(row["label"])
            file_path = train_root / relative_path
            if not file_path.is_file():
                raise FileNotFoundError(f"Missing file from split manifest: {file_path}")
            splits[split].append(DatasetItem(path=file_path, label=label, class_name=class_name))
    return splits


def ensure_split_manifest(
    *,
    train_root: Path = DEFAULT_TRAIN_DIR,
    manifest_path: Path | None = None,
    seed: int,
    validation_fraction: float = 0.2,
) -> Path:
    if manifest_path is None:
        manifest_path = DEFAULT_DATA_ROOT / "manifests" / "train_validation_split.csv"
    if not manifest_path.exists():
        return create_split_manifest(
            train_root=train_root,
            output_path=manifest_path,
            seed=seed,
            validation_fraction=validation_fraction,
        )
    return manifest_path


def build_dataset_manifest(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    dataset_slug: str = DATASET_SLUG,
    seed: int,
    output_path: Path,
    archive_path: Path | None = None,
) -> Path:
    train_summary = inspect_dataset(data_root / "Training")
    test_summary = inspect_dataset(data_root / "Testing")
    duplicate_report = build_duplicate_report(
        train_root=data_root / "Training",
        test_root=data_root / "Testing",
    )
    payload = {
        "manifest_version": 1,
        "dataset_slug": dataset_slug,
        "seed": seed,
        "train": train_summary,
        "test": test_summary,
        "duplicates": duplicate_report,
        "archive_path": str(archive_path) if archive_path else None,
        "archive_sha256": sha256_file(archive_path)
        if archive_path and archive_path.exists()
        else None,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def download_kaggle_dataset(
    *,
    data_root: Path = DEFAULT_DATA_ROOT,
    dataset_slug: str = DATASET_SLUG,
    manifest_path: Path | None = None,
    seed: int,
) -> tuple[Path, Path]:
    from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: PLC0415

    data_root.mkdir(parents=True, exist_ok=True)
    archive_path = data_root / "brain-tumor-classification.zip"

    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(data_root), quiet=False, unzip=False)

    if not archive_path.exists():
        downloaded_archives = sorted(data_root.glob("*.zip"))
        if not downloaded_archives:
            raise FileNotFoundError(
                "Kaggle download completed without a zip archive in the data directory."
            )
        archive_path = downloaded_archives[-1]

    with TemporaryDirectory() as temp_dir:
        extraction_root = Path(temp_dir) / "extract"
        extraction_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extraction_root)

        for folder_name in ("Training", "Testing"):
            source_dir = extraction_root / folder_name
            if not source_dir.exists():
                raise FileNotFoundError(f"Downloaded archive is missing {folder_name}")
            target_dir = data_root / folder_name
            if target_dir.exists():
                continue
            shutil.move(str(source_dir), str(target_dir))

    if manifest_path is None:
        manifest_path = Path("configs") / "brain-tumor.dataset.json"
    manifest_path = build_dataset_manifest(
        data_root=data_root,
        dataset_slug=dataset_slug,
        seed=seed,
        output_path=manifest_path,
        archive_path=archive_path,
    )
    return archive_path, manifest_path


def validate_expected_layout(data_root: Path = DEFAULT_DATA_ROOT) -> None:
    inspect_dataset(data_root / "Training")
    inspect_dataset(data_root / "Testing")
