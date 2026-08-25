from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from brain_tumor_classification.constants import CLASS_NAMES, DIRECTORY_NAMES

IMAGES_PER_CLASS = 3


@pytest.fixture()
def tiny_brain_dataset(tmp_path: Path) -> Path:
    data_root = tmp_path / "data"
    train_root = data_root / "Training"
    test_root = data_root / "Testing"
    predict_root = data_root / "New"
    predict_root.mkdir(parents=True, exist_ok=True)

    for root in (train_root, test_root):
        root_offset = 0 if root == train_root else 100
        for class_index, (class_name, directory_name) in enumerate(
            zip(CLASS_NAMES, DIRECTORY_NAMES, strict=True)
        ):
            class_dir = root / directory_name
            class_dir.mkdir(parents=True, exist_ok=True)
            for image_index in range(IMAGES_PER_CLASS):
                pixels = np.full(
                    (32, 32, 3),
                    fill_value=root_offset + (class_index * 40) + image_index,
                    dtype=np.uint8,
                )
                Image.fromarray(pixels).save(class_dir / f"{class_name}_{image_index}.png")

    Image.fromarray(np.full((32, 32, 3), 80, dtype=np.uint8)).save(predict_root / "sample.png")
    return data_root
